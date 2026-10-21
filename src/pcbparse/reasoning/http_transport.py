"""Bound an entire HTTP exchange, including DNS and address retries.

The worker uses only the standard library. Request credentials travel through
stdin; they never appear in process arguments, files or diagnostic messages.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


def request_json_with_deadline(request: urllib.request.Request, timeout: float) -> object:
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError('HTTP deadline exhausted')
    deadline = time.monotonic() + timeout
    envelope = json.dumps({
        'url': request.full_url, 'headers': dict(request.header_items()),
        'body': (request.data or b'').decode('utf-8'), 'timeout': timeout,
    }, ensure_ascii=False).encode('utf-8')
    if len(envelope) > MAX_REQUEST_BYTES:
        raise ValueError('HTTP request exceeds size limit')
    # -I/-S avoids importing the application, GPU runtime or user startup code.
    process = subprocess.Popen(
        [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
    )
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        output, _ = process.communicate(envelope, timeout=remaining)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError('HTTP total deadline exceeded') from None
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise
    if time.monotonic() > deadline:
        raise TimeoutError('HTTP total deadline exceeded')
    if process.returncode or len(output) > 2 * MAX_RESPONSE_BYTES:
        raise OSError('HTTP worker failed')
    try:
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError('Invalid HTTP worker response')
        if 'data' in result:
            return result['data']
        status = result.get('http_status')
        if type(status) is int and 100 <= status <= 599:
            raise urllib.error.HTTPError(request.full_url, status, 'HTTP request failed', None, None)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid HTTP worker response') from None
    raise urllib.error.URLError('HTTP transport failed')


def _worker() -> None:
    # Do not print exception messages: HTTP errors may contain sensitive bodies.
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError('oversized request')
        payload = json.loads(raw)
        request = urllib.request.Request(payload['url'],
            data=payload['body'].encode('utf-8'), headers=payload['headers'], method='POST')
        with urllib.request.urlopen(request, timeout=payload['timeout']) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError('oversized response')
        data = json.loads(body.decode('utf-8'))
        result = {'data': data}
    except urllib.error.HTTPError as exc:
        result = {'http_status': exc.code}
        exc.close()
    except Exception:
        result = {'error': 'transport_failed'}
    # Compact ASCII JSON bounds the worst-case escaping expansion.
    encoded = json.dumps(result, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if len(encoded) > 2 * MAX_RESPONSE_BYTES:
        encoded = b'{"error":"oversized_response"}'
    sys.stdout.buffer.write(encoded)


if __name__ == '__main__':
    _worker()
