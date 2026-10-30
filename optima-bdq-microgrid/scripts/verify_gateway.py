"""Verify an already deployed OPTIMA Basic Auth gateway from a test client."""
from __future__ import annotations

import base64
from http.cookies import SimpleCookie
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


TIMEOUT_SECONDS = 8
APPLICATION_ROUTES = ("/", "/_stcore/health")


class VerificationError(Exception):
    """A gateway verification assertion failed."""


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _request_status(
    url: str,
    authorization: str | None = None,
) -> tuple[int, str | None, str | None]:
    headers = {"User-Agent": "OPTIMA-gateway-verification/1"}
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            return (
                response.status,
                response.headers.get("Location"),
                response.headers.get("Set-Cookie"),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            error.headers.get("Location"),
            error.headers.get("Set-Cookie"),
        )


def _websocket_status(
    parsed_url: urllib.parse.SplitResult,
    authorization: str | None = None,
    xsrf_cookie: str | None = None,
) -> tuple[int, str]:
    port = parsed_url.port or 443
    raw_socket = socket.create_connection((parsed_url.hostname, port), TIMEOUT_SECONDS)
    context = ssl.create_default_context()
    with context.wrap_socket(raw_socket, server_hostname=parsed_url.hostname) as tls_socket:
        if tls_socket.version() not in ("TLSv1.2", "TLSv1.3"):
            raise VerificationError("Gateway did not negotiate TLS 1.2 or newer.")

        websocket_key = base64.b64encode(uuid.uuid4().bytes).decode("ascii")
        path = "/_stcore/stream"
        request_lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {parsed_url.netloc}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Version: 13",
            f"Sec-WebSocket-Key: {websocket_key}",
            f"Origin: https://{parsed_url.netloc}",
        ]
        if xsrf_cookie:
            request_lines.extend(
                (
                    f"Sec-WebSocket-Protocol: streamlit, {xsrf_cookie}",
                    f"Cookie: _streamlit_xsrf={xsrf_cookie}",
                )
            )
        if authorization:
            request_lines.append(f"Authorization: {authorization}")
        tls_socket.sendall(("\r\n".join(request_lines) + "\r\n\r\n").encode("ascii"))

        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = tls_socket.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > 16384:
                raise VerificationError("WebSocket response headers exceeded the limit.")

    first_line = bytes(response).split(b"\r\n", 1)[0].decode("ascii", errors="replace")
    parts = first_line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise VerificationError("Gateway returned an invalid WebSocket response.")
    return int(parts[1]), first_line


def _origin_is_reachable(origin_url: str) -> bool:
    parsed = urllib.parse.urlsplit(origin_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise VerificationError("OPTIMA_ORIGIN_URL must be an HTTP(S) origin URL.")
    if parsed.username or parsed.password:
        raise VerificationError("OPTIMA_ORIGIN_URL must not contain credentials.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        connection = socket.create_connection((parsed.hostname, port), TIMEOUT_SECONDS)
    except OSError:
        return False
    connection.close()
    return True


def _https_url(value: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise VerificationError("OPTIMA_GATEWAY_URL must be an HTTPS origin URL without credentials.")
    return parsed


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)
    print(f"PASS: {message}")


def main() -> int:
    gateway_url = os.environ.get("OPTIMA_GATEWAY_URL", "")
    username = os.environ.get("OPTIMA_TEST_USERNAME", "")
    password = os.environ.get("OPTIMA_TEST_PASSWORD", "")
    origin_url = os.environ.get("OPTIMA_ORIGIN_URL", "")
    if not all((gateway_url, username, password, origin_url)):
        print(
            "Set OPTIMA_GATEWAY_URL, OPTIMA_TEST_USERNAME, OPTIMA_TEST_PASSWORD, "
            "and OPTIMA_ORIGIN_URL in the test client environment.",
            file=sys.stderr,
        )
        return 2

    try:
        parsed = _https_url(gateway_url)
        authorization = "Basic " + base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")

        status, _, _ = _request_status(gateway_url)
        _check(status == 401, "Unauthenticated GET / is denied (401).")

        status, _, _ = _request_status(gateway_url, authorization)
        _check(status == 200, "Authenticated GET / is allowed (200).")

        xsrf_cookie = None
        for route in APPLICATION_ROUTES[1:]:
            status, _, _ = _request_status(urllib.parse.urljoin(gateway_url, route))
            _check(status == 401, f"Unauthenticated GET {route} is denied (401).")
            status, _, set_cookie = _request_status(
                urllib.parse.urljoin(gateway_url, route),
                authorization,
            )
            _check(status == 200, f"Authenticated GET {route} is allowed (200).")
            if route == "/_stcore/health" and set_cookie:
                cookies = SimpleCookie()
                cookies.load(set_cookie)
                if "_streamlit_xsrf" in cookies:
                    xsrf_cookie = cookies["_streamlit_xsrf"].value

        invalid_authorization = "Basic " + base64.b64encode(
            b"invalid-optima-user:invalid-optima-password"
        ).decode("ascii")
        status, _, _ = _request_status(gateway_url, invalid_authorization)
        _check(status == 401, "Invalid Basic Auth credentials are rejected (401).")

        status, response_line = _websocket_status(parsed)
        _check(status == 401, "Unauthenticated Streamlit WebSocket upgrade is denied (401).")
        if not xsrf_cookie:
            raise VerificationError("Authenticated health route did not provide the Streamlit XSRF cookie.")
        status, response_line = _websocket_status(parsed, authorization, xsrf_cookie)
        _check(status == 101, "Authenticated Streamlit WebSocket upgrade succeeds (101).")
        _check("101" in response_line, "Authenticated WebSocket returned Switching Protocols.")

        http_url = urllib.parse.urlunsplit(
            ("http", parsed.netloc, "/", "", "")
        )
        try:
            http_status, redirect, _ = _request_status(http_url)
            https_redirect = (
                http_status in (301, 302, 307, 308)
                and redirect is not None
                and urllib.parse.urlsplit(redirect).scheme == "https"
            )
            _check(https_redirect, "Cleartext HTTP is rejected or redirected to HTTPS.")
        except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError):
            _check(True, "Cleartext HTTP is rejected; the gateway requires HTTPS.")

        _check(
            not _origin_is_reachable(origin_url),
            "Direct Streamlit origin is unreachable from this test client's network.",
        )
    except (OSError, ssl.SSLError, VerificationError, urllib.error.URLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
