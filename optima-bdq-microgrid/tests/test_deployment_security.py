from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_streamlit_origin_is_loopback_only_with_browser_protections():
    config = _read(".streamlit/config.toml")
    compose = _read("compose.yaml")
    entrypoint = _read("deploy/entrypoint.sh")

    assert 'address = "127.0.0.1"' in config
    assert "enableCORS = true" in config
    assert "enableXsrfProtection = true" in config
    assert "gatherUsageStats = false" in config
    assert "STREAMLIT_SERVER_ADDRESS: 127.0.0.1" in compose
    assert "--server.address=127.0.0.1" in entrypoint
    assert "8501:8501" not in compose
    assert "127.0.0.1:8501" in entrypoint


def test_gateway_authenticates_all_http_and_websocket_routes():
    config = _read("deploy/nginx.conf.template")
    location = config.split("location / {", maxsplit=1)[1].split("}", maxsplit=1)[0]

    assert config.count("location ") == 1
    assert "auth_basic " in location
    assert "auth_basic_user_file /run/secrets/htpasswd;" in location
    assert "auth_basic off" not in config
    assert "proxy_http_version 1.1;" in location
    assert "proxy_set_header Upgrade $http_upgrade;" in location
    assert "proxy_set_header Connection $connection_upgrade;" in location
    assert "proxy_set_header Authorization \"\";" in location
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in location
    assert "proxy_set_header X-Forwarded-Proto https;" in location
    assert "proxy_set_header X-Forwarded-User \"\";" in location
    assert "proxy_set_header X-Auth-Request-User \"\";" in location


def test_gateway_requires_tls_and_publishes_only_loopback_gateway():
    config = _read("deploy/nginx.conf.template")
    compose = _read("compose.yaml")

    assert "listen 8443 ssl" in config
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in config
    assert "listen 80" not in config
    assert '"${GATEWAY_BIND_ADDRESS:-127.0.0.1}:${GATEWAY_PORT:-8443}:8443"' in compose
    assert "internal: true" in compose
    published_ports = compose.split("ports:", maxsplit=1)[1].split("secrets:", maxsplit=1)[0]
    assert "8501" not in published_ports


def test_gateway_fails_closed_without_external_auth_and_tls_secrets():
    compose = _read("compose.yaml")
    entrypoint = _read("deploy/entrypoint.sh")
    example = _read(".env.example")

    assert "${PUBLIC_HOST:?" in compose
    assert "${BASIC_AUTH_FILE:?" in compose
    assert "${TLS_CERT_FILE:?" in compose
    assert "${TLS_KEY_FILE:?" in compose
    assert "/run/secrets/htpasswd" in entrypoint
    assert "/run/secrets/tls.crt" in entrypoint
    assert "/run/secrets/tls.key" in entrypoint
    assert "nginx -t" in entrypoint
    assert "OPTIMA_TEST_PASSWORD" not in example
    assert "privkey.pem" in example


def test_gateway_verifier_covers_http_websocket_tls_and_origin():
    verifier = _read("scripts/verify_gateway.py")

    assert '"Unauthenticated GET / is denied (401)."' in verifier
    assert '"Authenticated GET / is allowed (200)."' in verifier
    assert '"Invalid Basic Auth credentials are rejected (401)."' in verifier
    assert '"Sec-WebSocket-Protocol: streamlit, {xsrf_cookie}"' in verifier
    assert '"Cookie: _streamlit_xsrf={xsrf_cookie}"' in verifier
    assert '"Authenticated Streamlit WebSocket upgrade succeeds (101)."' in verifier
    assert '"TLSv1.2", "TLSv1.3"' in verifier
    assert "_origin_is_reachable(origin_url)" in verifier
    assert "Cleartext HTTP is rejected" in verifier


def test_build_context_excludes_local_data_checkpoints_and_secrets():
    dockerignore = _read(".dockerignore")
    gitignore = _read(".gitignore")
    dockerfile = _read("deploy/Dockerfile")

    for excluded in ("data/", "models/", "results/", ".venv/", "env/", "deploy/secrets/"):
        assert excluded in dockerignore
    assert "deploy/secrets/" in gitignore
    assert "!.env.example" in gitignore
    assert "!.streamlit/config.toml" in dockerignore
    assert "COPY .streamlit/config.toml /opt/optima/.streamlit/config.toml" in dockerfile


def test_render_origin_uses_render_port_and_keeps_streamlit_loopback_only():
    nginx = _read("deploy/nginx-render.conf.template")
    entrypoint = _read("deploy/render-entrypoint.sh")
    dockerfile = _read("deploy/Dockerfile.render")

    assert "listen 0.0.0.0:${PORT} default_server;" in nginx
    assert "PORT=${PORT:-10000}" in entrypoint
    assert "envsubst '${PORT} ${BASIC_AUTH_FILE}'" in entrypoint
    assert "--server.address=127.0.0.1" in entrypoint
    assert "--server.port=8501" in entrypoint
    assert "EXPOSE 10000" in dockerfile
    assert "EXPOSE 8501" not in dockerfile
    assert "USER optima" in dockerfile
    assert "groupadd --gid 1000 render-secrets" in dockerfile
    for nginx_variable in (
        "$host",
        "$remote_addr",
        "$http_upgrade",
        "$connection_upgrade",
    ):
        assert nginx_variable in nginx


def test_render_auth_secret_uses_render_secret_file_and_fails_closed():
    entrypoint = _read("deploy/render-entrypoint.sh")
    nginx = _read("deploy/nginx-render.conf.template")
    env_example = _read("deploy/render.env.example")

    assert 'BASIC_AUTH_FILE=${BASIC_AUTH_FILE:-/etc/secrets/optima-htpasswd}' in entrypoint
    assert '[ ! -s "$BASIC_AUTH_FILE" ]' in entrypoint
    assert '[ ! -r "$BASIC_AUTH_FILE" ]' in entrypoint
    assert "Gateway startup refused" not in entrypoint
    assert "auth_basic " in nginx
    assert "auth_basic_user_file ${BASIC_AUTH_FILE};" in nginx
    assert "auth_basic off" not in nginx
    assert "BASIC_AUTH_FILE=/etc/secrets/optima-htpasswd" in env_example
    assert "tls.crt" not in entrypoint
    assert "tls.key" not in entrypoint


def test_render_default_server_protects_every_route_and_proxies_websockets():
    nginx = _read("deploy/nginx-render.conf.template")

    assert nginx.count("server {") == 1
    assert "default_server" in nginx
    assert "server_name _;" in nginx
    assert nginx.count("location / {") == 1
    assert "auth_basic " in nginx
    assert "auth_basic_user_file ${BASIC_AUTH_FILE};" in nginx
    assert "proxy_set_header Upgrade $http_upgrade;" in nginx
    assert "proxy_set_header Connection $connection_upgrade;" in nginx
    assert "proxy_set_header X-Forwarded-Proto https;" in nginx
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in nginx
    assert "proxy_set_header Authorization \"\";" in nginx


def test_render_dockerfile_and_build_context_include_only_render_config():
    dockerfile = _read("deploy/Dockerfile.render")
    dockerignore = _read(".dockerignore")

    assert "COPY deploy/nginx-render.conf.template" in dockerfile
    assert "COPY deploy/render-entrypoint.sh" in dockerfile
    assert "!deploy/Dockerfile.render" in dockerignore
    assert "!deploy/nginx-render.conf.template" in dockerignore
    assert "!deploy/render-entrypoint.sh" in dockerignore
    assert "deploy/secrets/" in dockerignore
    assert "data/" in dockerignore
    assert "models/" in dockerignore
