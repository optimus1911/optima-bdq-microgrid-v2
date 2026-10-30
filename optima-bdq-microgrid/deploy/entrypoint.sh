#!/bin/sh
set -eu

if [ -z "${PUBLIC_HOST:-}" ]; then
    echo "Gateway startup refused: PUBLIC_HOST is required." >&2
    exit 1
fi

if ! python -c 'import re,sys; sys.exit(0 if re.fullmatch(r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*", sys.argv[1]) else 1)' "$PUBLIC_HOST"; then
    echo "Gateway startup refused: PUBLIC_HOST must be a DNS hostname." >&2
    exit 1
fi

for secret in /run/secrets/htpasswd /run/secrets/tls.crt /run/secrets/tls.key; do
    if [ ! -s "$secret" ] || [ ! -r "$secret" ]; then
        echo "Gateway startup refused: a required authentication or TLS secret is unavailable." >&2
        exit 1
    fi
done

mkdir -p /tmp/nginx/client_temp /tmp/nginx/proxy_temp /tmp/home /tmp/cache /tmp/matplotlib
envsubst '${PUBLIC_HOST}' \
    < /opt/optima/deploy/nginx.conf.template \
    > /tmp/nginx.conf

if ! nginx -t -c /tmp/nginx.conf; then
    echo "Gateway startup refused: reverse-proxy configuration validation failed." >&2
    exit 1
fi

cd /opt/optima
python -m streamlit run app/dashboard.py \
    --server.address=127.0.0.1 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=true \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false \
    --client.showErrorDetails=none &
streamlit_pid=$!

attempt=0
until python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2).read()' >/dev/null 2>&1; do
    if ! kill -0 "$streamlit_pid" 2>/dev/null; then
        echo "Gateway startup refused: the loopback-only Streamlit origin did not start." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "Gateway startup refused: the loopback-only Streamlit origin did not become healthy." >&2
        kill -TERM "$streamlit_pid" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

nginx -g 'daemon off;' -c /tmp/nginx.conf &
nginx_pid=$!

cleanup() {
    kill -TERM "$nginx_pid" "$streamlit_pid" 2>/dev/null || true
    wait "$nginx_pid" 2>/dev/null || true
    wait "$streamlit_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

while kill -0 "$nginx_pid" 2>/dev/null && kill -0 "$streamlit_pid" 2>/dev/null; do
    sleep 1
done

echo "Gateway or Streamlit exited; stopping the deployment container." >&2
exit 1
