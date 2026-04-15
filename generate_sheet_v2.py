"""
OMR Answer Sheet v2 - clean bubble design (English + Arabic).
  - Letters sit BESIDE each bubble, not inside.
  - 4 question columns x 25 rows = 100 questions.
  - Large, bold question numbers.
  - Plain empty circles - OMR-friendly.

Usage:
  python generate_sheet_v2.py          # generate both EN + AR HTML
  python generate_sheet_v2.py --pdf    # also export PDFs (needs playwright)
"""

import argparse, base64, io, pathlib, qrcode

# ── language definitions ──────────────────────────────────────────────────────
LANGS = {
    "en": {
        "lang_attr": "en",
        "dir":       "ltr",
        "font":      "'Segoe UI', Arial, sans-serif",
        "title":     "Answer Sheet",
        "choices":   ["A", "B", "C", "D"],
        "lbl": {
            "student_name": "Student Name",
            "student_code": "Student Code",
            "exam_code":    "Exam Code",
            "class_name":   "Class",
            "student_qr":   "Student",
            "exam_qr":      "Exam",
            "legend":       "Legend:",
            "empty":        "Empty",
            "selected":     "Selected",
            "hint":         "Fill the bubble completely &nbsp;&middot;&nbsp; Use dark pencil or pen",
            "footer":       "OMR Answer Sheet &nbsp;|&nbsp; 100 Questions &nbsp;|&nbsp; A4",
        },
    },
    "ar": {
        "lang_attr": "ar",
        "dir":       "rtl",
        "font":      "'Noto Naskh Arabic', 'Segoe UI', Arial, sans-serif",
        "title":     "\u0648\u0631\u0642\u0629 \u0627\u0644\u0625\u062c\u0627\u0628\u0629",
        "choices":   ["\u0623", "\u0628", "\u062c", "\u062f"],
        "lbl": {
            "student_name": "\u0627\u0633\u0645 \u0627\u0644\u0637\u0627\u0644\u0628",
            "student_code": "\u0631\u0642\u0645 \u0627\u0644\u0637\u0627\u0644\u0628",
            "exam_code":    "\u0631\u0645\u0632 \u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646",
            "class_name":   "\u0627\u0644\u0641\u0635\u0644",
            "student_qr":   "\u0627\u0644\u0637\u0627\u0644\u0628",
            "exam_qr":      "\u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646",
            "legend":       "\u062f\u0644\u064a\u0644:",
            "empty":        "\u0641\u0627\u0631\u063a",
            "selected":     "\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0627\u0644\u0645\u062e\u062a\u0627\u0631\u0629",
            "hint":         "\u0627\u0645\u0644\u0623 \u0627\u0644\u062f\u0627\u0626\u0631\u0629 \u0643\u0627\u0645\u0644\u0627\u064b &nbsp;&middot;&nbsp; \u0627\u0633\u062a\u062e\u062f\u0645 \u0642\u0644\u0645\u0627\u064b \u063a\u0627\u0645\u0642\u0627\u064b",
            "footer":       "\u0648\u0631\u0642\u0629 \u0625\u062c\u0627\u0628\u0629 OMR &nbsp;|&nbsp; 100 \u0633\u0624\u0627\u0644 &nbsp;|&nbsp; A4",
        },
    },
}

# ── shared data ───────────────────────────────────────────────────────────────
DATA = {
    "student_name": "Ahmed Ali",
    "student_code": "STU-20240001",
    "exam_code":    "EXAM-2024-MATH-01",
    "class_name":   "Grade 10-A",
    "n_questions":  100,
}

# ── QR helper ─────────────────────────────────────────────────────────────────
def qr_b64(text: str) -> str:
    qr = qrcode.QRCode(version=None,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=6, border=2)
    qr.add_data(text); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── question grid ─────────────────────────────────────────────────────────────
def build_grid(n: int, choices: list, rtl: bool) -> str:
    rpc = (n + 3) // 4          # rows per column
    cols = []
    for col in range(4):
        start = col * rpc + 1
        end   = min((col + 1) * rpc, n)
        rows  = []
        for q in range(start, end + 1):
            # For RTL: bubble THEN label (reads right-to-left: label sits to the left of bubble)
            # For LTR: bubble THEN label (label sits to the right of bubble)
            bubble_pairs = "".join(
                f'<span class="bw">'
                f'<span class="bubble"></span>'
                f'<span class="bl">{ch}</span>'
                f'</span>'
                for ch in choices
            )
            rows.append(
                f'<div class="qr">'
                f'<span class="qn">{q}</span>'
                f'<span class="bs">{bubble_pairs}</span>'
                f'</div>'
            )
        cols.append('<div class="qc">' + "".join(rows) + "</div>")
    return '<div class="qg">' + "".join(cols) + "</div>"

# ── full HTML ─────────────────────────────────────────────────────────────────
def generate_html(lang_key: str, data: dict) -> str:
    L          = LANGS[lang_key]
    lbl        = L["lbl"]
    d          = L["dir"]
    rtl        = (d == "rtl")
    choices    = L["choices"]
    student_qr = qr_b64(data["student_code"])
    exam_qr    = qr_b64(data["exam_code"])
    grid       = build_grid(data["n_questions"], choices, rtl)

    # legend sample: one empty + one filled
    legend_sample = (
        f'<span class="bw"><span class="bubble"></span>'
        f'<span class="bl">= {lbl["empty"]}</span></span>'
        f'&nbsp;&nbsp;'
        f'<span class="bw"><span class="bubble bubble-filled"></span>'
        f'<span class="bl">= {lbl["selected"]}</span></span>'
    )

    return f"""<!DOCTYPE html>
<html lang="{L['lang_attr']}" dir="{d}">
<head>
<meta charset="UTF-8"/>
<title>{L['title']}</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}

  body {{
    font-family: {L['font']};
    font-size: 11px;
    background: #bbb;
    color: #111;
    direction: {d};
  }}

  /* ── A4 sheet ── */
  .sheet {{
    position: relative;
    width: 210mm;
    height: 297mm;
    margin: 0 auto;
    background: #fff;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ── Corner fiducials ── */
  .anchor {{
    position: absolute;
    width: 11mm; height: 11mm;
    background: #000;
    z-index: 10;
  }}
  .anchor.tl {{ top:5mm;    left:5mm;   }}
  .anchor.tr {{ top:5mm;    right:5mm;  }}
  .anchor.bl {{ bottom:5mm; left:5mm;   }}
  .anchor.br {{ bottom:5mm; right:5mm;  }}

  /* ── Content ── */
  .content {{
    flex: 1;
    padding: 14mm 20mm 8mm;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}

  /* ── Title ── */
  .title {{
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 1px;
    border-bottom: 3px double #222;
    padding-bottom: 6px;
  }}

  /* ── Header ── */
  .header {{
    display: flex;
    gap: 10px;
    align-items: stretch;
  }}

  .meta {{ flex:1; display:flex; flex-direction:column; gap:5px; }}

  .meta-row {{ display:flex; gap:10px; }}

  .field {{
    flex:1;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }}
  .field.wide {{ flex:2; }}

  .field label {{
    font-size: 7.5px;
    font-weight: 700;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}

  .field .val {{
    border-bottom: 1.5px solid #333;
    padding: 2px 4px;
    font-size: 12px;
    font-weight: 600;
    min-height: 19px;
    line-height: 1.4;
  }}

  /* ── QR section ── */
  .qrs {{ display:flex; gap:8px; align-items:center; }}

  .qrbox {{
    display:flex; flex-direction:column; align-items:center; gap:2px;
  }}
  .qrbox img  {{ width:17mm; height:17mm; display:block; }}
  .qrbox span {{
    font-size:7.5px; color:#555; font-weight:700;
    text-transform:uppercase; letter-spacing:0.3px;
  }}

  /* ── Legend / instructions bar ── */
  .legend {{
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    font-size: 9px;
    border-top: 1.5px solid #bbb;
    border-bottom: 1.5px solid #bbb;
    padding: 3px 0;
    background: #fafafa;
  }}
  .legend .ltitle {{
    font-weight: 800;
    font-size: 9.5px;
  }}
  .legend .hint {{
    color: #555;
    margin-{"left" if not rtl else "right"}: auto;
    font-size: 8.5px;
  }}

  /* ── Question grid ── */
  .qg {{
    flex: 1;
    display: flex;
    gap: 0;
    align-items: flex-start;
  }}

  .qc {{
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 0 3px;
    border-{"right" if not rtl else "left"}: 1px dashed #ddd;
  }}
  .qc:last-child {{
    border-{"right" if not rtl else "left"}: none;
  }}

  /* ── Single question row ── */
  .qr {{
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 1.6px 0;
    border-bottom: 1px solid #f0f0f0;
  }}

  .qn {{
    font-size: 11.5px;
    font-weight: 800;
    min-width: 23px;
    text-align: {"right" if not rtl else "left"};
    color: #111;
    flex-shrink: 0;
    line-height: 1;
  }}

  .bs {{
    display: flex;
    gap: 3px;
    align-items: center;
    flex-wrap: nowrap;
  }}

  /* ── Bubble + label pair ── */
  .bw {{
    display: flex;
    align-items: center;
    gap: 2px;
    flex-shrink: 0;
  }}

  .bubble {{
    display: inline-block;
    width:  14px;
    height: 14px;
    border-radius: 50%;
    border: 1.8px solid #111;
    background: #fff;
    flex-shrink: 0;
  }}

  .bubble-filled {{
    background: #111;
  }}

  .bl {{
    font-size: 9.5px;
    font-weight: 700;
    color: #222;
    line-height: 1;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    font-size: 8px;
    color: #888;
    border-top: 1px solid #ccc;
    padding: 3px 0;
  }}

  @media print {{
    body {{ background:none; }}
    .sheet {{ margin:0; }}
  }}
</style>
</head>
<body>
<div class="sheet">
  <div class="anchor tl"></div>
  <div class="anchor tr"></div>
  <div class="anchor bl"></div>
  <div class="anchor br"></div>

  <div class="content">

    <div class="title">{L['title']}</div>

    <div class="header">
      <div class="meta">
        <div class="meta-row">
          <div class="field wide">
            <label>{lbl['student_name']}</label>
            <div class="val">{data['student_name']}</div>
          </div>
          <div class="field">
            <label>{lbl['student_code']}</label>
            <div class="val">{data['student_code']}</div>
          </div>
        </div>
        <div class="meta-row">
          <div class="field">
            <label>{lbl['exam_code']}</label>
            <div class="val">{data['exam_code']}</div>
          </div>
          <div class="field">
            <label>{lbl['class_name']}</label>
            <div class="val">{data['class_name']}</div>
          </div>
        </div>
      </div>

      <div class="qrs">
        <div class="qrbox">
          <img src="data:image/png;base64,{student_qr}" alt="Student QR"/>
          <span>{lbl['student_qr']}</span>
        </div>
        <div class="qrbox">
          <img src="data:image/png;base64,{exam_qr}" alt="Exam QR"/>
          <span>{lbl['exam_qr']}</span>
        </div>
      </div>
    </div>

    <div class="legend">
      <span class="ltitle">{lbl['legend']}</span>
      {legend_sample}
      <span class="hint">{lbl['hint']}</span>
    </div>

    {grid}

    <div class="footer">{lbl['footer']}</div>

  </div>
</div>
</body>
</html>
"""

# ── PDF via Playwright ────────────────────────────────────────────────────────
def to_pdf(html_path: str, pdf_path: str) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page()
        page.goto(f"file:///{pathlib.Path(html_path).resolve()}")
        page.pdf(path=pdf_path, format="A4", print_background=True,
                 margin={"top":"0","bottom":"0","left":"0","right":"0"})
        browser.close()
    print(f"[PDF] saved -> {pdf_path}")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="Also export PDFs (needs playwright)")
    args = ap.parse_args()

    for lang in ("en", "ar"):
        stem = f"answer_sheet_v2_{lang}"
        html = generate_html(lang, DATA)
        pathlib.Path(f"{stem}.html").write_text(html, encoding="utf-8")
        print(f"[{lang.upper()}] HTML saved -> {stem}.html")
        if args.pdf:
            to_pdf(f"{stem}.html", f"{stem}.pdf")
