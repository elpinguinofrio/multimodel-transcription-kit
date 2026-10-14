"""Multipart transcription against an OpenAI-compatible endpoint (e.g. local Parakeet).

``transcribe`` returns the parsed response and goes through the shared limiter
and retry policy; ``run`` also writes request/response receipts for benchmarks.
"""
import argparse
import hashlib
import json
import mimetypes
import time
import urllib.error
import uuid
from pathlib import Path

import requests

from .errors import ProviderError, ResponseError
from .request_policy import send_with_limits
from .usage import UsageLog

ACCEPTED_SUFFIXES = {'.wav', '.m4a', '.mp3', '.flac', '.aiff'}


class HTTPProviderError(ProviderError, urllib.error.HTTPError):
    """Non-200 from the endpoint. Also an HTTPError, as run() raised before 0.3."""

    def __init__(self, url, code, reason):
        urllib.error.HTTPError.__init__(self, url, code, reason, {}, None)
        ProviderError.__init__(self, f'HTTP {code} from transcription endpoint: {reason}')
        self.status = code

    def __str__(self):
        return self.args[0]

def multipart(audio, model, language, boundary, prompt=None):
    pieces = []
    for key, value in [('model', model), ('language', language), ('response_format', 'verbose_json'),
                       ('prompt', prompt)]:
        if value is None:
            continue  # An omitted language lets the provider auto-detect it.
        if '\r' in value or '\n' in value:
            raise ValueError('Multipart values must be single-line')
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    content_type = mimetypes.guess_type(audio.name)[0] or 'application/octet-stream'
    # Fixed safe filename avoids injecting the caller filename into MIME headers.
    pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio{audio.suffix}"\r\nContent-Type: {content_type}\r\n\r\n'.encode())
    pieces.extend([audio.read_bytes(), f'\r\n--{boundary}--\r\n'.encode()])
    return b''.join(pieces)

def transcribe(audio, endpoint, *, model='parakeet-tdt-0.6b-v3', language=None, prompt=None,
               timeout=300, limiter=None, usage_log=None, max_retries=4, base_delay=2.0,
               max_delay=60.0):
    """POST one file; return (parsed response, raw bytes, http status, seconds).

    429/5xx/connection errors are retried under the shared policy; redirects are
    never followed. Raises HTTPProviderError on a non-200, ResponseError when the
    body has no string text.
    """
    audio = Path(audio)
    if audio.suffix.lower() not in ACCEPTED_SUFFIXES:
        raise ValueError('Unsupported audio suffix')
    if not endpoint.startswith(('http://', 'https://')):
        raise ValueError('An explicit HTTP(S) endpoint is required')
    if prompt is not None and ('\r' in prompt or '\n' in prompt):
        prompt = ' '.join(prompt.split())
    boundary = 'benchmark-' + uuid.uuid4().hex
    body = multipart(audio, model, language, boundary, prompt=prompt)
    headers = {'Content-Type': 'multipart/form-data; boundary=' + boundary}
    started = time.perf_counter()
    response = send_with_limits(
        lambda: requests.post(endpoint, data=body, headers=headers, timeout=timeout, allow_redirects=False),
        limiter=limiter, key=model, max_retries=max_retries, base_delay=base_delay,
        max_delay=max_delay, usage_log=usage_log, provider='http', operation='transcription')
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        # The body is not echoed: servers can include private diagnostics.
        raise HTTPProviderError(endpoint, response.status_code, response.reason or 'error')
    raw = response.content
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ResponseError(f'Response is not JSON: {exc}') from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get('text'), str):
        raise ResponseError('Response lacks string text')
    return parsed, raw, response.status_code, elapsed


def run(audio, endpoint, output, model='parakeet-tdt-0.6b-v3', language='ru', timeout=300,
        usage_log=None, limiter=None):
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    parsed, raw, http_status, elapsed = transcribe(audio, endpoint, model=model, language=language,
                                                   timeout=timeout, limiter=limiter, usage_log=usage_log)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'response.json').write_bytes(raw)
    metadata = dict(endpoint=endpoint, audio_name=audio.name, audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), requested_model=model, language=language, response_format='verbose_json', http_status=http_status, request_wall_seconds=elapsed, recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), model_identity='requested_not_runtime_verified')
    (output / 'request.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='parakeet-tdt-0.6b-v3')
    parser.add_argument('--language', default='ru')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--usage-file', type=Path, help='Persistent local request usage SQLite file')
    args = parser.parse_args(argv)
    log = UsageLog(args.usage_file) if args.usage_file else None
    print(json.dumps(run(args.audio, args.endpoint, args.output, args.model, args.language,
                         args.timeout, usage_log=log), indent=2))

if __name__ == '__main__':
    main()
