#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
DEFAULT_CERT_FILE="${DEMO_DIR}/certs/cert.pem"
DEFAULT_KEY_FILE="${DEMO_DIR}/certs/key.pem"

PUBLIC_HOST="${MINICPMO_PUBLIC_HOST:-localhost}"
EXTRA_DNS="${MINICPMO_TLS_EXTRA_DNS:-}"
EXTRA_IPS="${MINICPMO_TLS_EXTRA_IPS:-}"
CERT_FILE="${MINICPMO_SSL_CERTFILE:-${DEFAULT_CERT_FILE}}"
KEY_FILE="${MINICPMO_SSL_KEYFILE:-${DEFAULT_KEY_FILE}}"
DAYS="${MINICPMO_TLS_DAYS:-365}"
FORCE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/generate_minicpmo45_lan_cert.sh [options]

Generate a self-signed TLS certificate for MiniCPM-o LAN access.

Options:
  --public-host HOST     Hostname or IP users open in the browser. Default: localhost
  --tls-extra-dns LIST   Comma-separated extra DNS SAN entries
  --tls-extra-ip LIST    Comma-separated extra IP SAN entries
  --certfile PATH        Output certificate path. Default: third_party/.../certs/cert.pem
  --keyfile PATH         Output private key path. Default: third_party/.../certs/key.pem
  --days N               Certificate validity days. Default: 365
  --force                Overwrite existing output files
  -h, --help             Show this help

Environment overrides:
  MINICPMO_PUBLIC_HOST
  MINICPMO_TLS_EXTRA_DNS
  MINICPMO_TLS_EXTRA_IPS
  MINICPMO_SSL_CERTFILE
  MINICPMO_SSL_KEYFILE
  MINICPMO_TLS_DAYS
EOF
}

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${ROOT_DIR}/${value}"
  fi
}

is_ip_literal() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ || "${value}" == *:* ]]
}

append_unique_csv() {
  local current="$1"
  local candidate="$2"
  local item
  IFS=',' read -r -a items <<< "${current}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${candidate}" ]]; then
      printf '%s\n' "${current}"
      return 0
    fi
  done
  if [[ -z "${current}" ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s,%s\n' "${current}" "${candidate}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --public-host)
      PUBLIC_HOST="$2"
      shift 2
      ;;
    --tls-extra-dns)
      EXTRA_DNS="$2"
      shift 2
      ;;
    --tls-extra-ip|--tls-extra-ips)
      EXTRA_IPS="$2"
      shift 2
      ;;
    --certfile)
      CERT_FILE="$2"
      shift 2
      ;;
    --keyfile)
      KEY_FILE="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate TLS certificates" >&2
  exit 1
fi

if [[ -z "${PUBLIC_HOST}" ]]; then
  echo "--public-host must not be empty" >&2
  exit 1
fi

CERT_FILE="$(resolve_path "${CERT_FILE}")"
KEY_FILE="$(resolve_path "${KEY_FILE}")"

if [[ "${FORCE}" != "1" && ( -e "${CERT_FILE}" || -e "${KEY_FILE}" ) ]]; then
  echo "Refusing to overwrite existing certificate files without --force" >&2
  echo "  cert: ${CERT_FILE}" >&2
  echo "  key : ${KEY_FILE}" >&2
  exit 1
fi

dns_entries="localhost"
ip_entries="127.0.0.1"

if is_ip_literal "${PUBLIC_HOST}"; then
  ip_entries="$(append_unique_csv "${ip_entries}" "${PUBLIC_HOST}")"
else
  dns_entries="$(append_unique_csv "${dns_entries}" "${PUBLIC_HOST}")"
fi

if [[ -n "${EXTRA_DNS}" ]]; then
  IFS=',' read -r -a extra_dns_items <<< "${EXTRA_DNS}"
  for item in "${extra_dns_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    dns_entries="$(append_unique_csv "${dns_entries}" "${item}")"
  done
fi

if [[ -n "${EXTRA_IPS}" ]]; then
  IFS=',' read -r -a extra_ip_items <<< "${EXTRA_IPS}"
  for item in "${extra_ip_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    ip_entries="$(append_unique_csv "${ip_entries}" "${item}")"
  done
fi

mkdir -p "$(dirname "${CERT_FILE}")" "$(dirname "${KEY_FILE}")"

tmp_conf="$(mktemp)"
trap 'rm -f "${tmp_conf}"' EXIT

{
  printf '[req]\n'
  printf 'distinguished_name = req_dn\n'
  printf 'x509_extensions = v3_req\n'
  printf 'prompt = no\n'
  printf '\n[req_dn]\n'
  printf 'CN = %s\n' "${PUBLIC_HOST}"
  printf '\n[v3_req]\n'
  printf 'subjectAltName = @alt_names\n'
  printf '\n[alt_names]\n'

  dns_idx=1
  IFS=',' read -r -a dns_items <<< "${dns_entries}"
  for item in "${dns_items[@]}"; do
    printf 'DNS.%s = %s\n' "${dns_idx}" "${item}"
    dns_idx=$((dns_idx + 1))
  done

  ip_idx=1
  IFS=',' read -r -a ip_items <<< "${ip_entries}"
  for item in "${ip_items[@]}"; do
    printf 'IP.%s = %s\n' "${ip_idx}" "${item}"
    ip_idx=$((ip_idx + 1))
  done
} > "${tmp_conf}"

openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
  -days "${DAYS}" \
  -keyout "${KEY_FILE}" \
  -out "${CERT_FILE}" \
  -config "${tmp_conf}" \
  -extensions v3_req >/dev/null 2>&1

echo "Generated MiniCPM-o TLS certificate:"
echo "  cert: ${CERT_FILE}"
echo "  key : ${KEY_FILE}"
echo "  DNS SANs: ${dns_entries}"
echo "  IP  SANs: ${ip_entries}"
