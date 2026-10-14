"""
STT API service to handle speech-to-text conversion using Google Gemini.
"""

import io
import logging
import os
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
import importlib.util

from .rate_limiter import estimate_audio_tokens, estimate_text_tokens
from .request_policy import send_with_limits

logger = logging.getLogger("stt_api_gemini")


def _load_audio_converter():
    from . import audio_converter
    return audio_converter


class GeminiModel(Enum):
    """Available Gemini models for speech recognition.

    RULE: never list models below 3.x here. Older generations have been
    proven unreliable on these accounts and are intentionally excluded.
    """
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_5_TRANSCRIBE = "gemini-3.5-transcribe"
    GEMINI_3_5_TRANSCRIBE_LIVE = "gemini-3.5-transcribe-live"
    GEMINI_3_8_LIVE = "gemini-3.8-live"
    # Legacy alias kept ONLY because old benchmark filenames reference it.
    # Do not use for new transcription work.
    GEMINI_PRO_VISION = "gemini-pro-vision"


# Minimum acceptable Gemini major version. Anything below this raises.
MIN_MODEL_MAJOR_VERSION = 3


def _model_major_version(model_name: str) -> Optional[int]:
    """Extract the major version digit from a model name like 'gemini-3.5-flash'."""
    import re
    m = re.match(r"gemini-(\d+)", model_name or "")
    return int(m.group(1)) if m else None


def assert_model_allowed(model_name: str) -> None:
    """Reject any Gemini model below the configured minimum major version."""
    major = _model_major_version(model_name)
    if major is None:
        return  # non-versioned names (e.g. pro-vision alias) skip the check
    if major < MIN_MODEL_MAJOR_VERSION:
        raise GeminiAPIError(
            f"Refusing to use {model_name}: below minimum major version "
            f"{MIN_MODEL_MAJOR_VERSION}. Pick a gemini-3.x model."
        )


class GeminiAPIError(Exception):
    """Raised when the Gemini API client encounters a fatal error."""


class GeminiAPI:
    """Client for Google Gemini API speech-to-text capabilities.

    Auth is an API key (GEMINI_API_KEY) against a configurable endpoint:
    GEMINI_BASE_URL overrides the host, defaulting to Google's public API.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    VALID_FORMATS = {"json", "text"}

    # Transient statuses worth retrying: 429 rate-limit + 5xx upstream blips.
    # 4xx (400/401/403/404) are permanent — retrying only wastes the credit.
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    # ───────────────────────── init ──────────────────────────

    def __init__(
        self,
        throttle_rate: float = 0.5,
        timeout: int = 300,
        requests_per_minute: int = 10,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        max_retry_delay: float = 60.0,
        rate_limiter=None,
    ) -> None:
        """
        Initialize the Gemini API client.

        Args:
            throttle_rate: Minimum time between API calls in seconds.
            timeout: Base API request timeout in seconds.
            requests_per_minute: Per-minute request cap.
            rate_limiter: Shared RateLimiter; when set it replaces the built-in throttle.
        """
        # .env must load unconditionally first — a shell-exported GEMINI_API_KEY
        # would otherwise short-circuit loading and hide GEMINI_BASE_URL.
        self._load_env_files()
        self.api_key: Optional[str] = self._get_api_key()
        # GEMINI_BASE_URL overrides the endpoint host (e.g. an alternate
        # Gemini-compatible gateway); defaults to Google's public API.
        self.base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        self.API_URL = f"{self.base_url}/v1beta/models/{{model}}:generateContent"
        self.throttle_rate = throttle_rate
        self.timeout = timeout
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_retry_delay = max_retry_delay
        self.rate_limiter = rate_limiter
        self._last_call = 0.0
        # Track request timestamps for rate limiting (sliding window)
        self._request_timestamps: deque = deque()
        logger.info(
            "GeminiAPI initialized throttle=%s timeout=%s rpm=%d",
            throttle_rate, timeout, requests_per_minute,
        )

    @staticmethod
    def _load_env_files() -> None:
        """Load only the explicitly selected TRANSCRIPTION_ENV_FILE."""
        from .config import load_env_file
        load_env_file()

    @staticmethod
    def _get_api_key() -> str:
        from .config import require_key
        return require_key("GEMINI_API_KEY", GeminiAPIError)

    # ───────────────────────── helpers ──────────────────────────

    def _throttle(self) -> None:
        """Enforce minimum time between API calls to prevent rate limiting."""
        delta = time.time() - self._last_call
        if delta < self.throttle_rate:
            time.sleep(self.throttle_rate - delta)
        self._last_call = time.time()

    def _rate_limit(self) -> None:
        """
        Enforce requests per minute rate limiting.
        Ensures requests are spaced evenly (e.g., 6 seconds apart for 10 req/min).
        """
        current_time = time.time()
        seconds_per_request = 60.0 / self.requests_per_minute

        # Remove timestamps older than 1 minute
        one_minute_ago = current_time - 60.0
        while self._request_timestamps and self._request_timestamps[0] < one_minute_ago:
            self._request_timestamps.popleft()

        # If we have recent requests, ensure minimum spacing
        if self._request_timestamps:
            # Get the most recent request time
            last_request_time = self._request_timestamps[-1]
            time_since_last = current_time - last_request_time

            # If not enough time has passed, wait
            if time_since_last < seconds_per_request:
                wait_time = seconds_per_request - time_since_last + 0.1  # Add 0.1s buffer
                logger.info("Rate limiting: waiting %.1f seconds (%.1fs per request for %d req/min)...",
                           wait_time, seconds_per_request, self.requests_per_minute)
                time.sleep(wait_time)
                current_time = time.time()
                # Clean up old timestamps after waiting
                one_minute_ago = current_time - 60.0
                while self._request_timestamps and self._request_timestamps[0] < one_minute_ago:
                    self._request_timestamps.popleft()

        # Record this request timestamp
        self._request_timestamps.append(current_time)

    def _post_with_retry(self, url, payload, timeout, headers=None, limit_key=None, tokens=0,
                         max_retries=None):
        """POST with bounded retry on transient failures.

        Retries on 429 / 5xx (see RETRYABLE_STATUS) and on connection-level
        errors (resets, read timeouts), up to max_retries, honoring Retry-After.
        Permanent 4xx are returned as-is for the caller to raise on. This is the
        single network exit for transcription — a transient 429 on one chunk of a
        long recording used to raise immediately and burn that segment into a hole.
        With a rate_limiter, every attempt first waits for a free slot under
        `limit_key` (the model name).
        """
        return send_with_limits(
            lambda: requests.post(url, json=payload, timeout=timeout, headers=headers or None),
            limiter=self.rate_limiter, key=limit_key or "gemini", tokens=tokens,
            max_retries=self.max_retries if max_retries is None else max_retries,
            base_delay=self.retry_base_delay, max_delay=self.max_retry_delay,
        )

    @staticmethod
    def _encode_audio_base64(audio_bytes: bytes) -> str:
        """
        Encode audio bytes to base64 for API request.

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            Base64 encoded string
        """
        import base64
        return base64.b64encode(audio_bytes).decode("utf-8")

    @staticmethod
    def _get_audio_duration(file_path: str) -> Optional[float]:
        """
        Get audio duration in seconds using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds, or None if ffprobe is unavailable or parsing fails
        """
        import subprocess
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    # ───────────────────────── hallucination cleaning ───────────────────────

    def _clean_hallucinations(self, text: str) -> str:
        """
        Detect and remove hallucinated repetition loops at the end of text.
        Loops until all patterns removed. Returns [HALLUCINATION] if entire text is garbage.
        """
        if not text or len(text) < 50:
            return text

        # Loop until no more patterns found
        iterations = 0
        while iterations < 10:  # Safety limit
            cleaned = self._clean_one_hallucination_pattern(text)
            if cleaned == text:
                break
            if cleaned == "[HALLUCINATION]":
                return cleaned
            text = cleaned
            iterations += 1

        return text

    def _clean_one_hallucination_pattern(self, text: str) -> str:
        """Remove one repetition pattern from end of text."""
        if len(text) < 50:
            return text

        # Strategy: Find the smallest repeating UNIT at the end
        # Check lengths from 5 to 100 characters (reasonable phrase length)
        for length in range(5, min(100, len(text) // 4)):
            suffix = text[-length:]
            # Check in a window of 5x the pattern length
            scan_size = min(len(text), length * 5)
            scan_window = text[-scan_size:]

            count = scan_window.count(suffix)
            if count >= 4:  # Pattern repeats 4+ times
                # Found the repeating unit - now find where it starts
                pattern_start = self._find_repetition_start(text, suffix)

                if pattern_start is not None:
                    if pattern_start < 50:
                        # Entire text is basically repetition
                        logger.warning(f"Entire chunk is repetition of '{suffix[:30]}...'")
                        return "[HALLUCINATION]"

                    logger.warning(f"Detected loop (length {length}): '{suffix[:20]}...'. Truncating at char {pattern_start}.")
                    return text[:pattern_start]

        return text

    def _find_repetition_start(self, text: str, pattern: str) -> int:
        """
        Find where the CONTIGUOUS repetition block starts at the end of text.

        Searches backwards to avoid truncating legitimate earlier occurrences.
        Only identifies where the dense, back-to-back repetition begins.

        Returns:
            Position where contiguous repetition block starts, or None if not found.
        """
        if not pattern or pattern not in text:
            return None

        # Find ALL occurrences of the pattern
        positions = []
        start = 0
        while True:
            idx = text.find(pattern, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

        if len(positions) < 4:
            return None

        # Work backwards from the end to find where contiguous block starts
        # Contiguous = gap between occurrences is less than 2x pattern length
        max_gap = len(pattern) * 2

        # Start from second-to-last position, check gaps going backwards
        contiguous_start_idx = len(positions) - 1  # Index in positions list

        for i in range(len(positions) - 2, -1, -1):
            gap = positions[i + 1] - positions[i] - len(pattern)
            if gap > max_gap:
                # Found a large gap - block starts AFTER this position
                contiguous_start_idx = i + 1
                break
            else:
                contiguous_start_idx = i

        block_start = positions[contiguous_start_idx]

        # Check if there's meaningful content before the block
        prefix = text[:block_start].strip()
        if len(prefix) < 30:
            # Very little real content - treat as all garbage
            return 0

        return block_start

    def _strip_conversational_prefix(self, text: str) -> str:
        """
        Remove conversational artifacts that LLMs sometimes add.
        E.g., "Answer:", "Here is the transcription:", etc.
        """
        if not text:
            return text

        # Common prefixes the LLM adds before transcription
        prefixes_to_strip = [
            "answer:",
            "here is the transcription:",
            "here's the transcription:",
            "transcription:",
            "the transcription is:",
            "the audio says:",
        ]

        text_lower = text.lower().strip()
        for prefix in prefixes_to_strip:
            if text_lower.startswith(prefix):
                stripped = text[len(prefix):].strip()
                if stripped:  # Only strip if there's content after
                    logger.debug(f"Stripped conversational prefix: '{prefix}'")
                    return stripped

        return text

    # ───────────────────────── public api ───────────────────────

    def list_available_models(self) -> list:
        """
        List all available models from the Gemini API.

        Returns:
            List of model names that are available
        """
        list_url = f"{self.base_url}/v1beta/models?key={self.api_key}"

        try:
            res = requests.get(list_url, timeout=self.timeout)
            if res.status_code != 200:
                logger.warning("Failed to list models: %s", res.text)
                return []

            models_data = res.json()
            available_models = []

            if "models" in models_data:
                for model in models_data["models"]:
                    model_name = model.get("name", "")
                    # Extract just the model name (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
                    if "/" in model_name:
                        model_name = model_name.split("/")[-1]

                    # Check if model supports generateContent
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" in supported_methods:
                        available_models.append(model_name)

            logger.info("Found %d available models: %s", len(available_models), available_models)
            return available_models

        except Exception as e:
            logger.warning("Error listing models: %s", e)
            return []

    # Valid output formats for transcription style
    VALID_OUTPUT_FORMATS = {"default", "court", "screenplay"}

    def transcribe_live(self, audio_path, model):
        """Transcribe a recording over Live using this client's configuration."""
        from .gemini_live import transcribe_live
        return transcribe_live(audio_path, model, api=self)

    def speech_to_text(
        self,
        audio_input: Union[str, bytes],
        model: Union[GeminiModel, str] = GeminiModel.GEMINI_3_FLASH_PREVIEW,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        diarize: bool = False,
        analyze: bool = False,
        output_format: str = "default",
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Gemini API.

        Args:
            audio_input: Path to audio file or raw audio bytes
            model: Gemini model to use for transcription (enum or string)
            language: Language code (e.g., 'en', 'ru') or None for auto-detect
            prompt: Optional prompt to guide transcription
            response_format: Output format ('text' or 'json')
            temperature: Sampling temperature (1.0 recommended for Gemini 3)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            diarize: If True, identify and label different speakers
            analyze: If True, extract full audio metadata (emotions, gender, etc.)
            output_format: Transcription style ('default', 'court', 'screenplay')

        Returns:
            Dictionary containing transcription and metadata
        """
        if response_format not in self.VALID_FORMATS:
            raise ValueError(f"response_format must be one of {self.VALID_FORMATS}")
        if output_format not in self.VALID_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {self.VALID_OUTPUT_FORMATS}")

        # Get model name (handle both enum and string)
        if isinstance(model, GeminiModel):
            model_name = model.value
        else:
            model_name = str(model)

        from .gemini_live import LIVE_MODELS
        if model_name in LIVE_MODELS:
            if diarize or analyze or output_format != "default" or language or prompt:
                raise ValueError("Live models support plain transcription only; use transcribe_live")
            if not isinstance(audio_input, (str, Path)):
                raise ValueError("Live transcription requires an audio file path")
            return self.transcribe_live(audio_input, model_name)

        # Handle audio conversion if needed
        temp_mp3_path = None
        audio_path = None
        mime_type = "audio/m4a"  # Default MIME type

        if isinstance(audio_input, str):
            audio_path = audio_input
            filename = Path(audio_input).name

            # Check if conversion is needed
            _audio_conv = _load_audio_converter()
            AudioConverter = _audio_conv.AudioConverter
            AudioFormat = _audio_conv.AudioFormat

            audio_format = AudioFormat.from_file_extension(audio_path)

            # Convert video/CAF files to MP3 - REQUIRED for API compatibility
            # Gemini API rejects raw MOV/MP4/CAF with refusals, so fail fast if we can't convert
            if audio_format in (AudioFormat.CAF, AudioFormat.MOV, AudioFormat.MP4):
                if not AudioConverter.check_ffmpeg_installed():
                    raise GeminiAPIError(
                        f"ffmpeg required for {audio_format.value} files but not installed. "
                        f"Install ffmpeg or convert file to MP3/M4A/WAV first."
                    )
                logger.info(f"{audio_format.value} file detected - converting to MP3 (required for API)")
                try:
                    temp_mp3_path = AudioConverter.convert_to_mp3(audio_path, bitrate="64k")
                    audio_path = temp_mp3_path
                    mime_type = "audio/mpeg"
                    logger.info(f"Converted {audio_format.value} to MP3")
                except Exception as e:
                    raise GeminiAPIError(
                        f"Failed to convert {audio_format.value} to MP3: {e}. "
                        f"File may be corrupted or ffmpeg encountered an error."
                    ) from e
            elif audio_format == AudioFormat.MP3:
                mime_type = "audio/mpeg"
            elif audio_format == AudioFormat.M4A:
                mime_type = "audio/m4a"
            elif audio_format == AudioFormat.WAV:
                mime_type = "audio/wav"

            # Get audio bytes from (possibly converted) file
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            # Get actual duration using ffprobe if available
            duration_sec = self._get_audio_duration(audio_path)
            if duration_sec is None:
                # Fallback to rough estimate
                duration_sec = len(audio_bytes) / 10000
        else:
            audio_bytes = audio_input
            logger.info("Using raw audio bytes (length=%d)", len(audio_bytes))
            # Rough estimate for raw bytes
            duration_sec = len(audio_bytes) / 10000

        # Encode audio to base64
        audio_b64 = self._encode_audio_base64(audio_bytes)

        # Prepare prompt text based on output_format or mode flags
        if output_format == "court":
            prompt_text = (
                f"Transcribe the following audio in COURT REPORTER format. "
                f"Use this exact structure:\n\n"
                f"1. Start with a header block:\n"
                f"   TRANSCRIPTION\n"
                f"   [Date if mentioned]\n"
                f"   SPEAKERS: List each speaker as 'SPEAKER N (description) - Gender'\n\n"
                f"2. Number each line of dialogue starting from 1\n"
                f"3. Use ALL CAPS for speaker labels: 'SPEAKER 1:', 'SPEAKER 2:'\n"
                f"4. Use parentheticals for non-speech events: (Laughter), (Pause), (Inaudible), (Crosstalk)\n"
                f"5. Use '--' for interrupted speech\n"
                f"6. Mark unclear words as (inaudible) or (unintelligible)\n\n"
                f"Example:\n"
                f"1    SPEAKER 1: Can you hear me?\n"
                f"2    SPEAKER 2: Yeah, loud and clear.\n"
                f"3              (Laughter)\n"
                f"4    SPEAKER 1: So about the refactor --\n"
                f"5    SPEAKER 2: Go ahead.\n\n"
                f"If the audio is silent or empty, output exactly '[SILENCE]'. "
                f"Do NOT ask for the audio file. Do NOT converse. "
                f"{prompt or ''}"
            )
        elif output_format == "screenplay":
            prompt_text = (
                f"Transcribe the following audio in SCREENPLAY format. "
                f"Use this exact structure:\n\n"
                f"1. Character names in ALL CAPS on their own line\n"
                f"2. Dialogue indented below the character name\n"
                f"3. Parentheticals for emotions/delivery in lowercase: (hesitant), (excited), (whispering)\n"
                f"4. Action/sound descriptions in parentheses: (laughter), (phone rings), (door slams)\n"
                f"5. Scene descriptions for audio environment changes: (traffic noise in background)\n\n"
                f"Example:\n"
                f"(quiet room, keyboard clicking)\n\n"
                f"SPEAKER 1\n"
                f"Can you hear me?\n\n"
                f"SPEAKER 2\n"
                f"Yeah, loud and clear.\n\n"
                f"(both laugh)\n\n"
                f"SPEAKER 1\n"
                f"(hesitant)\n"
                f"So about the refactor...\n\n"
                f"If the audio is silent or empty, output exactly '[SILENCE]'. "
                f"Do NOT ask for the audio file. Do NOT converse. "
                f"{prompt or ''}"
            )
        elif analyze:
            prompt_text = (
                f"Analyze the following audio and provide a comprehensive report. "
                f"Include ALL of the following:\n\n"
                f"## SPEAKERS\n"
                f"For each speaker detected:\n"
                f"- Label (Speaker 1, Speaker 2, etc.)\n"
                f"- Estimated gender (male/female/uncertain)\n"
                f"- Estimated age range (child/young adult/adult/elderly)\n"
                f"- Primary language(s) spoken\n"
                f"- Emotional tone throughout (happy, sad, frustrated, neutral, excited, etc.)\n"
                f"- Speech characteristics (fast/slow, clear/mumbled, confident/hesitant)\n\n"
                f"## TRANSCRIPTION\n"
                f"Full transcription with speaker labels. Format: 'Speaker N: [text]'\n\n"
                f"## AUDIO ENVIRONMENT\n"
                f"- Background sounds detected (music, traffic, nature, silence, etc.)\n"
                f"- Audio quality assessment (clear/noisy/distorted)\n"
                f"- Notable non-speech sounds (laughter, sighs, coughs, etc.)\n\n"
                f"## SUMMARY\n"
                f"- Brief summary of what this recording is about\n"
                f"- Overall mood/atmosphere of the conversation\n\n"
                f"If the audio is silent or empty, output exactly '[SILENCE]'. "
                f"Do NOT ask for the audio file. Do NOT converse. "
                f"{prompt or ''}"
            )
        elif diarize:
            prompt_text = (
                f"Transcribe the following audio with speaker diarization. "
                f"Identify different speakers and label them as 'Speaker 1:', 'Speaker 2:', etc. "
                f"Format each speaker's turn on a new line with their label. "
                f"Do not include timestamps or time markers. "
                f"If the audio is silent, empty, or unclear, output exactly '[SILENCE]'. "
                f"Do NOT ask for the audio file. Do NOT converse. "
                f"{prompt or ''}"
            )
        else:
            prompt_text = (
                f"Transcribe the following audio to plain text. "
                f"Do not include timestamps or time markers. "
                f"Output only the spoken words. "
                f"If the audio is silent, empty, or unclear, output exactly '[SILENCE]'. "
                f"Do NOT ask for the audio file. Do NOT converse. "
                f"{prompt or ''}"
            )
        if language:
            prompt_text += f" The audio is in {language} language."

        # Validate model is allowed (rejects pre-3.x families).
        assert_model_allowed(model_name)

        # Endpoint + auth: API key in the query string against API_URL
        # (Google's public API, or whatever GEMINI_BASE_URL points at).
        request_headers: Dict[str, str] = {}
        request_url = self.API_URL.format(model=model_name)
        request_url += f"?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt_text
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k
            }
        }

        # Built-in spacing only when no shared limiter is managing this model.
        if self.rate_limiter is None:
            self._throttle()  # Minimum time between requests
            self._rate_limit()  # Requests per minute limit

        # Calculate dynamic timeout based on audio duration
        # Base timeout + 30 seconds per minute of audio (free tier can be slow)
        # Minimum 300s, maximum 900s (15 min)
        duration_minutes = duration_sec / 60
        dynamic_timeout = max(
            self.timeout,
            min(900, 300 + int(duration_minutes * 30))  # 30s per minute of audio
        )
        if dynamic_timeout > self.timeout:
            logger.info("Using extended timeout: %ds (audio duration: %.1f min)",
                       dynamic_timeout, duration_minutes)

        # Make the API request
        logger.info("Sending request to Gemini API (model=%s)", model_name)
        started = time.time()

        try:
            res = self._post_with_retry(
                request_url,
                payload,
                timeout=dynamic_timeout,
                headers=request_headers or None,
                limit_key=model_name,
                tokens=estimate_audio_tokens(duration_sec),
            )

            if res.status_code != 200:
                try:
                    detail = res.json()
                except Exception:
                    detail = res.text
                logger.error("API error %d: %s", res.status_code, detail)
                raise GeminiAPIError(f"API error {res.status_code}: {detail}")

            processing_time = time.time() - started
            logger.info("API response received in %.2f seconds", processing_time)

            response_data = res.json()

            # Extract text from the response
            try:
                parts = response_data["candidates"][0]["content"]["parts"]
                texts = [
                    part.get("text", part.get("audioTranscription", {}).get("text"))
                    for part in parts if not part.get("thought")
                ]
                texts = [value for value in texts if isinstance(value, str)]
                if not texts:
                    raise KeyError("text or audioTranscription.text")
                text = "".join(texts)
            except (KeyError, IndexError) as e:
                # Log detailed response structure for debugging
                import json
                logger.error("Failed to extract text from response: %s", e)

                # Extract what we can from the response for detailed error
                candidates = response_data.get("candidates", [])
                if candidates:
                    candidate = candidates[0]
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    safety_ratings = candidate.get("safetyRatings", [])
                    content = candidate.get("content", {})

                    # Build detailed error message
                    error_details = f"finishReason={finish_reason}"
                    if safety_ratings:
                        blocked = [r for r in safety_ratings if r.get("blocked")]
                        if blocked:
                            error_details += f", BLOCKED by: {[r.get('category') for r in blocked]}"
                    if not content:
                        error_details += ", NO CONTENT returned"
                    elif not content.get("parts"):
                        error_details += f", content has no parts: {list(content.keys())}"

                    logger.error("Response details: %s", error_details)
                    logger.debug("Full candidate: %s", json.dumps(candidate, indent=2)[:1000])
                    text = f"[PARSE_ERROR] {error_details}"
                else:
                    # No candidates at all
                    prompt_feedback = response_data.get("promptFeedback", {})
                    block_reason = prompt_feedback.get("blockReason", "")
                    if block_reason:
                        error_details = f"PROMPT BLOCKED: {block_reason}"
                    else:
                        error_details = f"No candidates returned, keys: {list(response_data.keys())}"
                    logger.error("Response details: %s", error_details)
                    logger.debug("Full response: %s", json.dumps(response_data, indent=2)[:1000])
                    text = f"[PARSE_ERROR] {error_details}"

            # Clean hallucinations/repetition loops
            text = self._clean_hallucinations(text)

            # Strip conversational artifacts (LLM sometimes adds these)
            text = self._strip_conversational_prefix(text)

            # Extract usage metadata for stats
            usage_metadata = response_data.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)
            output_bytes = len(text.encode('utf-8'))

            # Format duration as minutes:seconds (e.g., 1465 seconds = 24m25s)
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            duration_str = f"{minutes}m{seconds}s"

            # Format tokens with "k" suffix if >= 1000 (e.g., 15200 = 15.2k)
            def format_tokens(tokens):
                if tokens >= 1000:
                    return f"{tokens / 1000:.1f}k"
                return str(tokens)

            input_tokens_str = format_tokens(input_tokens)
            output_tokens_str = format_tokens(output_tokens)

            # Format bytes to kb with one decimal place (e.g., 3700 bytes = 3.7kb)
            output_kb = output_bytes / 1024
            output_bytes_str = f"{output_kb:.1f}kb"

            # Log stats in requested format
            logger.info("input: %s/%s tokens, output %s/%s tokens",
                       duration_str, input_tokens_str, output_bytes_str, output_tokens_str)

            # Build result dictionary (return token stats for caller logging)
            result = {
                "text": text,
                "duration": duration_sec,
                "processing_time": processing_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model_name,
                "success": True,
            }

            if response_format == "json":
                # Include more details in JSON format
                result["raw_response"] = response_data

            return result

        except requests.exceptions.RequestException as exc:
            logger.error("Request failed: %s", exc)
            raise GeminiAPIError(f"request failed: {exc}") from exc
        finally:
            # Clean up temporary MP3 file if created
            if temp_mp3_path and Path(temp_mp3_path).exists():
                try:
                    os.unlink(temp_mp3_path)
                    logger.debug(f"Removed temporary MP3 file: {temp_mp3_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove temporary MP3 file {temp_mp3_path}: {e}")

    # ───────────────────────── text generation ──────────────────────────

    def generate_text(
        self,
        prompt: str,
        model: Union[GeminiModel, str] = "gemini-3.5-flash",
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        retries: int = 4,
    ) -> str:
        """Text-only generateContent against the configured endpoint (the local Gemini proxy
        when GEMINI_BASE_URL is set). Returns the model's text. Raises GeminiAPIError on a
        non-200 response or an unparseable/blocked candidate — callers surface an honest
        empty-state rather than fabricating output. No STT hallucination-cleaners here.
        """
        if not self.api_key:
            raise GeminiAPIError("GEMINI_API_KEY not set — cannot call the LLM")
        model_name = model.value if isinstance(model, GeminiModel) else model

        request_url = self.API_URL.format(model=model_name) + f"?key={self.api_key}"
        gen_config: Dict[str, object] = {"temperature": temperature}
        if max_output_tokens is not None:
            gen_config["maxOutputTokens"] = max_output_tokens
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }

        if self.rate_limiter is None:
            self._throttle()
            self._rate_limit()
        # 429 and 5xx are retried with back-off rather than failing the batch.
        try:
            res = self._post_with_retry(request_url, payload, timeout=timeout or self.timeout,
                                        limit_key=model_name, tokens=estimate_text_tokens(prompt),
                                        max_retries=retries)
        except requests.exceptions.RequestException as exc:
            raise GeminiAPIError(f"request failed: {exc}") from exc

        if res.status_code != 200:
            try:
                detail = res.json()
            except Exception:
                detail = res.text
            raise GeminiAPIError(f"API error {res.status_code}: {detail}")

        try:
            data = res.json()
        except ValueError as exc:
            raise GeminiAPIError(f"200 but body was not valid JSON: {exc}") from exc
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            d = data if isinstance(data, dict) else {}
            cands = d.get("candidates") or []
            cand = cands[0] if isinstance(cands, list) and cands and isinstance(cands[0], dict) else {}
            reason = cand.get("finishReason", "UNKNOWN")
            pf = d.get("promptFeedback")
            block = pf.get("blockReason", "") if isinstance(pf, dict) else ""
            raise GeminiAPIError(
                f"no usable text (finishReason={reason}"
                f"{', blockReason=' + block if block else ''})"
            ) from exc
