#!/bin/sh
set -eu

PORT=${PORT:-10000}
BASIC_AUTH_FILE=${BASIC_AUTH_FILE:-/etc/secrets/optima-htpasswd}

case "$PORT" in
    ''|*[!0-9]*)
        echo "Render startup refused: PORT must be a valid TCP port." >&2
        exit 1
        ;;
esac
if ! python -c 'import sys; p=int(sys.argv[1]); sys.exit(0 if 1 <= p <= 65535 else 1)' "$PORT"; then
    echo "Render startup refused: PORT must be between 1 and 65535." >&2
    exit 1
fi

case "$BASIC_AUTH_FILE" in
    /*) ;;
    *)
        echo "Render startup refused: BASIC_AUTH_FILE must be an absolute path." >&2
        exit 1
        ;;
esac
case "$BASIC_AUTH_FILE" in
    *[!A-Za-z0-9_./-]*)
        echo "Render startup refused: BASIC_AUTH_FILE contains unsupported path characters." >&2
        exit 1
        ;;
esac
if [ ! -s "$BASIC_AUTH_FILE" ] || [ ! -r "$BASIC_AUTH_FILE" ]; then
    echo "Render startup refused: the required Basic Auth Secret File is unavailable." >&2
    exit 1
fi

mkdir -p /tmp/nginx/client_temp /tmp/nginx/proxy_temp /tmp/home /tmp/cache /tmp/matplotlib
envsubst '${PORT} ${BASIC_AUTH_FILE}' \
    < /opt/optima/deploy/nginx-render.conf.template \
    > /tmp/nginx.conf

if ! nginx -t -c /tmp/nginx.conf; then
    echo "Render startup refused: reverse-proxy configuration validation failed." >&2
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
        echo "Render startup refused: the loopback-only Streamlit origin did not start." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "Render startup refused: the loopback-only Streamlit origin did not become healthy." >&2
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

echo "Render gateway or Streamlit exited; stopping the deployment container." >&2
exit 1
