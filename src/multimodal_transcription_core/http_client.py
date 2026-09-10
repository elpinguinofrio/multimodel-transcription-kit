"""Run one explicit multipart transcription request; never called by offline CI."""
import argparse
import hashlib
import json
import mimetypes
import time
import urllib.request
import uuid
from pathlib import Path

def multipart(audio, model, language, boundary):
    pieces = []
    for key, value in [('model', model), ('language', language), ('response_format', 'verbose_json')]:
        if '\r' in value or '\n' in value:
            raise ValueError('Multipart values must be single-line')
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    content_type = mimetypes.guess_type(audio.name)[0] or 'application/octet-stream'
    # Fixed safe filename avoids injecting the caller filename into MIME headers.
    pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio{audio.suffix}"\r\nContent-Type: {content_type}\r\n\r\n'.encode())
    pieces.extend([audio.read_bytes(), f'\r\n--{boundary}--\r\n'.encode()])
    return b''.join(pieces)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def run(audio, endpoint, output, model='parakeet-tdt-0.6b-v3', language='ru', timeout=300):
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if audio.suffix.lower() not in {'.wav', '.m4a', '.mp3', '.flac', '.aiff'}:
        raise ValueError('Unsupported audio suffix')
    if not endpoint.startswith(('http://', 'https://')):
        raise ValueError('An explicit HTTP(S) endpoint is required')
    boundary = 'benchmark-' + uuid.uuid4().hex
    body = multipart(audio, model, language, boundary)
    request = urllib.request.Request(endpoint, data=body, headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}, method='POST')
    started = time.perf_counter()
    with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
        raw = response.read()
        http_status = response.status
    elapsed = time.perf_counter() - started
    parsed = json.loads(raw)
    if not isinstance(parsed.get('text'), str):
        raise ValueError('Response lacks string text')
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
    args = parser.parse_args(argv)
    print(json.dumps(run(args.audio, args.endpoint, args.output, args.model, args.language, args.timeout), indent=2))

if __name__ == '__main__':
    main()
