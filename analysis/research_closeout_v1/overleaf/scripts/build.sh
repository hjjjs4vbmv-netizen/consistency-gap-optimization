#!/usr/bin/env bash
set -euo pipefail
ECT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ECT_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ect-en-closeout.XXXXXX")"
cd "$ECT_PROJECT_DIR"
mkdir -p compiled
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$ECT_BUILD_DIR" main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates -o "$ECT_BUILD_DIR" main.tex
else
  printf 'Install latexmk/pdfLaTeX or Tectonic.\n' >&2
  exit 1
fi
cp "$ECT_BUILD_DIR/main.pdf" compiled/ECT_Draft_EN.pdf.tmp
mv compiled/ECT_Draft_EN.pdf.tmp compiled/ECT_Draft_EN.pdf
printf 'English PDF: %s\nBuild logs: %s\n' "$ECT_PROJECT_DIR/compiled/ECT_Draft_EN.pdf" "$ECT_BUILD_DIR"
