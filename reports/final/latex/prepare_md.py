"""Chuẩn bị file markdown của báo cáo cho pandoc.

Ghi phần thân vào stdout và phần thông tin đầu báo cáo (đề tài, thành viên, ghi chú) vào file thứ hai,
để đặt phần này ở trang bìa. Đồng thời bỏ mục lục viết tay, đường kẻ ngang, emoji, và thêm dòng trống
trước các danh sách mà pandoc không nhận ra.
"""
import re
import sys

src, front_out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

# Phần đầu: từ sau tiêu đề (dòng "# ...") tới trước "## Tóm tắt".
head, body = text.split("\n## Tóm tắt", 1)
title_line, front = head.split("\n", 1)
front = re.sub(r"^---\s*$", "", front, flags=re.M)
# Mỗi dòng **Nhãn:** là một dòng riêng trên trang bìa.
front = re.sub(r"^(\*\*[^*]+:\*\*.*)$", r"| \1", front, flags=re.M)
open(front_out, "w", encoding="utf-8").write(front)
text = title_line + "\n\n## Tóm tắt" + body

# Mục lục viết tay: LaTeX tự sinh mục lục.
text = re.sub(r"## Mục lục\n.*?(?=\n## )", "", text, flags=re.S)
# Đường kẻ ngang giữa các mục: mỗi mục đã sang trang mới.
text = re.sub(r"^---\s*$", "", text, flags=re.M)
# Danh sách viết liền sau một đoạn văn cần một dòng trống phía trước.
text = re.sub(r"(?m)^((?![ \t]*(?:[-*]|\d+\.)\s)(?!\|)\S.*)\n(?=(?:\d+\.|-) )", r"\1\n\n", text)
replacements = {
    "📁 ": "Nguồn: ", "✅": "Đạt", "⚠️": "Một phần", "❌": "Chưa",
    "✔ một phần": "Một phần", "✔": "Có", "10¹⁰": "10^10^",
}
for old, new in replacements.items():
    text = text.replace(old, new)
sys.stdout.write(text)
