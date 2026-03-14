#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/.local_duplex"
LATEST_FILE="${RUNTIME_DIR}/latest_session"
TARGET="${1:-}"

if [ -z "${TARGET}" ]; then
  if [ ! -f "${LATEST_FILE}" ]; then
    echo "No local duplex session summary found." >&2
    exit 1
  fi
  TARGET="$(cat "${LATEST_FILE}")"
fi

if [ -d "${TARGET}" ]; then
  SUMMARY_FILE="${TARGET}/interaction.md"
elif [ -f "${TARGET}" ]; then
  SUMMARY_FILE="${TARGET}"
else
  echo "Session path not found: ${TARGET}" >&2
  exit 1
fi

if [ ! -f "${SUMMARY_FILE}" ]; then
  echo "Interaction summary not found: ${SUMMARY_FILE}" >&2
  exit 1
fi

cat "${SUMMARY_FILE}"
