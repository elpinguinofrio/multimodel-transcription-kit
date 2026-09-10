"""
Audio conversion service that handles audio format conversion using ffmpeg.
"""

import importlib.util
import logging
import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional


def _load_logger():
    return logging


_logger_module = _load_logger()
logger = _logger_module.getLogger("audio_converter")


class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = ".wav"
    M4A = ".m4a"
    CAF = ".caf"
    MOV = ".mov"
    MP3 = ".mp3"
    MP4 = ".mp4"

    @classmethod
    def from_file_extension(cls, file_path: str) -> Optional["AudioFormat"]:
        """Get AudioFormat enum from file extension"""
        ext = Path(file_path).suffix.lower()
        for fmt in cls:
            if fmt.value == ext:
                return fmt
        return None


class AudioConverter:
    """Utility class to convert audio files between formats using ffmpeg"""

    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if ffmpeg is installed and available"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @staticmethod
    def convert_to_mp3(input_path: str, bitrate: str = "64k") -> str:
        """
        Convert audio file to MP3 format with compression to reduce file size.
        Useful for large CAF files before further processing.

        Args:
            input_path: Path to input audio file
            bitrate: MP3 bitrate (default: 64k for speech, can use 128k for better quality)

        Returns:
            Path to converted MP3 file (temporary file)
        """
        if not AudioConverter.check_ffmpeg_installed():
            raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to convert audio files.")

        # Create a temporary file for the MP3 output
        temp_dir = tempfile.gettempdir()
        input_filename = Path(input_path).stem
        output_path = Path(temp_dir) / f"{input_filename}_compressed.mp3"

        logger.info(f"Converting {input_path} to MP3 format (bitrate: {bitrate}) at {output_path}")

        try:
            # Run ffmpeg to convert to MP3 with compression
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-ar", "16000",  # Sample rate (16kHz) - sufficient for speech
                "-ac", "1",      # Mono
                "-b:a", bitrate, # Bitrate for compression
                "-y",            # Overwrite output file if it exists
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            logger.debug(f"ffmpeg MP3 conversion completed: {result.stderr}")
            return str(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting audio to MP3: {e.stderr}")
            raise RuntimeError(f"Failed to convert audio file to MP3: {e.stderr}")

    @staticmethod
    def convert_to_wav(input_path: str) -> str:
        """
        Convert audio file to WAV format

        Args:
            input_path: Path to input audio file

        Returns:
            Path to converted WAV file (temporary file)
        """
        if not AudioConverter.check_ffmpeg_installed():
            raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to convert audio files.")

        # Create a temporary file for the WAV output
        temp_dir = tempfile.gettempdir()
        input_filename = Path(input_path).stem
        output_path = Path(temp_dir) / f"{input_filename}_converted.wav"

        logger.info(f"Converting {input_path} to WAV format at {output_path}")

        try:
            # Run ffmpeg to convert the file
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-ar", "16000",  # Sample rate (16kHz)
                "-ac", "1",      # Mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                "-y",            # Overwrite output file if it exists
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            logger.debug(f"ffmpeg conversion completed: {result.stderr}")
            return str(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting audio: {e.stderr}")
            raise RuntimeError(f"Failed to convert audio file: {e.stderr}")

    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """
        Get audio duration in seconds using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds, or None if unable to determine
        """
        if not AudioConverter.check_ffmpeg_installed():
            return None
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Could not get audio duration: {e}")
            return None

    @staticmethod
    def get_metadata(file_path: str) -> Optional[dict]:
        """
        Get audio metadata (duration and creation date) in a single ffprobe call.

        Args:
            file_path: Path to audio file

        Returns:
            Dict with 'duration' (float) and 'creation_date' (datetime or None),
            or None if ffprobe fails entirely.
        """
        from datetime import datetime, timezone

        if not AudioConverter.check_ffmpeg_installed():
            return None

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration:format_tags=creation_time",
                    "-of", "default=noprint_wrappers=0",
                    file_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            output = result.stdout
            duration = None
            creation_date = None

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('duration='):
                    try:
                        duration = float(line.split('=')[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('TAG:creation_time=') or line.startswith('creation_time='):
                    try:
                        date_str = line.split('=', 1)[1]
                        # Parse ISO format (UTC), handle various formats
                        date_str = date_str.replace('Z', '+00:00')
                        if '.' in date_str:
                            # Has microseconds
                            creation_date = datetime.fromisoformat(date_str.split('.')[0]).replace(tzinfo=timezone.utc)
                        else:
                            creation_date = datetime.fromisoformat(date_str)
                        # Convert to local time
                        creation_date = creation_date.astimezone()
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Could not parse creation_time: {e}")

            # Fallback to file mtime if no creation_time in metadata
            if creation_date is None:
                try:
                    mtime = os.path.getmtime(file_path)
                    creation_date = datetime.fromtimestamp(mtime).astimezone()  # Timezone-aware
                except OSError:
                    pass

            return {
                "duration": duration,
                "creation_date": creation_date
            }

        except Exception as e:
            logger.debug(f"Could not get audio metadata: {e}")
            return None
