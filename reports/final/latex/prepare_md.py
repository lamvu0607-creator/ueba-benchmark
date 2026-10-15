"""Chuẩn bị file markdown của báo cáo cho pandoc.

Bỏ dòng tiêu đề (khối tiêu đề nằm trong build.sh), đặt mục lục ngay sau mục Phân công như trang 1
của các báo cáo tuần, bỏ mục lục viết tay, đường kẻ ngang, emoji, và thêm dòng trống trước các danh
sách mà pandoc không nhận ra.
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
text = text.split("\n", 1)[1]  # dòng "# Báo cáo kỹ thuật: ..."
text = text.replace("\n## Tóm tắt", "\n```{=latex}\n\\tableofcontents\n```\n\n## Tóm tắt", 1)
text = re.sub(r"## Mục lục\n.*?(?=\n## )", "", text, flags=re.S)
text = re.sub(r"^---\s*$", "", text, flags=re.M)
text = re.sub(r"(?m)^((?![ \t]*(?:[-*]|\d+\.)\s)(?!\|)\S.*)\n(?=(?:\d+\.|-) )", r"\1\n\n", text)
replacements = {
    "📁 ": "Nguồn: ", "✅": "Đạt", "⚠️": "Một phần", "❌": "Chưa",
    "✔ một phần": "Một phần", "✔": "Có", "10¹⁰": "10^10^", "└─": "+-",
}
for old, new in replacements.items():
    text = text.replace(old, new)
sys.stdout.write(text)
