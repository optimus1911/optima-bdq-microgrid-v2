FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gettext-base nginx \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 optima \
    && useradd --system --uid 10001 --gid 10001 --home-dir /tmp/home optima

WORKDIR /opt/optima

COPY requirements.txt /opt/optima/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY configs/default.yaml /opt/optima/configs/default.yaml
COPY src/optima/ /opt/optima/src/optima/
COPY app/ /opt/optima/app/
COPY .streamlit/config.toml /opt/optima/.streamlit/config.toml
COPY deploy/nginx.conf.template /opt/optima/deploy/nginx.conf.template
COPY deploy/entrypoint.sh /opt/optima/deploy/entrypoint.sh

RUN mkdir -p /tmp/nginx/client_temp /tmp/nginx/proxy_temp /tmp/home /tmp/cache /tmp/matplotlib \
    && chown -R 10001:10001 /opt/optima /tmp/nginx /tmp/home /tmp/cache /tmp/matplotlib \
    && chmod 0555 /opt/optima/deploy/entrypoint.sh

USER 10001:10001
EXPOSE 8443
ENTRYPOINT ["/opt/optima/deploy/entrypoint.sh"]
