#!/usr/bin/env bash
# Chuyển reports/final/bao_cao_ky_thuat.md sang LaTeX rồi compile ra PDF.
# Trình bày theo kiểu báo cáo tuần (pdflatex + vntex, Computer Modern tiếng Việt).
# Cần: pandoc >= 3.1, pdflatex (TeX Live: vntex, fvextra, titlesec, booktabs, newunicodechar).
# Dùng: ./build.sh            -> bao_cao_ky_thuat.tex và bao_cao_ky_thuat.pdf
#       ./build.sh --tex-only -> chỉ sinh bao_cao_ky_thuat.tex
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare_md.py ../bao_cao_ky_thuat.md front_input.md > build_input.md
{
  cat <<'TEX'
\begin{center}
{\color{tieude}\LARGE\bfseries BÁO CÁO KỸ THUẬT – UEBA BENCHMARK}\\[3pt]
{\large Benchmark Isolation Forest, LOF, One-Class SVM trên log xác thực Windows (Event 4624/4625)}\\[3pt]
\textbf{Thành viên:} Lâm Vũ – Đinh Xuân Hiệp \quad | \quad Báo cáo tổng tuần 1–6, ngày 15/10/2026
\end{center}
\vspace{2pt}
TEX
  pandoc front_input.md -f markdown --lua-filter=md2tex.lua -t latex
  echo '\vspace{4pt}'
} > front.tex
pandoc build_input.md -f markdown-implicit_figures+pipe_tables \
  --shift-heading-level-by=-1 --toc --toc-depth=2 \
  -M pagetitle="Báo cáo kỹ thuật – UEBA Benchmark" \
  -V documentclass=article -V fontsize=11pt -V colorlinks=true -V linkcolor=black \
  -H header.tex -B front.tex --lua-filter=md2tex.lua --highlight-style=tango \
  --standalone -o bao_cao_ky_thuat.tex
rm -f build_input.md front_input.md front.tex
if [ "${1:-}" != "--tex-only" ]; then
  for _ in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error bao_cao_ky_thuat.tex > /dev/null
  done
  rm -f bao_cao_ky_thuat.{aux,log,out,toc}
  echo "Đã tạo $(pwd)/bao_cao_ky_thuat.pdf"
fi
