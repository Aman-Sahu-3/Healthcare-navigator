# syntax=docker/dockerfile:1.7
#
# Healthcare Knowledge Navigator — web image
#   build stage: Node compiles the React app to static files
#   runtime:     nginx serves them and proxies /api/* to the API container

# ---------- build ----------
FROM node:22-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
# Empty = call the API through this server's /api proxy (same origin, no CORS).
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ---------- runtime ----------
FROM nginx:1.29-alpine AS runtime

COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
# hadolint ignore=DL3025
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
