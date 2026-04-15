"""
OMR Answer Sheet v3
  - 5 choices (A B C D E / ا ب ج د ه)
  - Large bubbles, large numbers, large header
  - Letters as column header above each group (not beside each bubble)
  - 4 corner anchors + 2 mid-edge + column-top/bottom markers for robust OMR

Usage:
  python generate_sheet_v3.py
  python generate_sheet_v3.py --pdf   (needs playwright)
"""

import argparse, base64, io, pathlib, qrcode

# ── language definitions ──────────────────────────────────────────────────────
LANGS = {
    "en": {
        "lang_attr": "en",
        "dir":       "ltr",
        "font":      "'Segoe UI', Arial, sans-serif",
        "title":     "Answer Sheet",
        "choices":   ["A", "B", "C", "D", "E"],
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
            "hint":         "Fill bubble completely with dark pencil or pen",
            "footer":       "OMR Answer Sheet  |  100 Questions  |  5 Choices  |  A4",
        },
    },
    "ar": {
        "lang_attr": "ar",
        "dir":       "rtl",
        "font":      "'Noto Naskh Arabic', 'Segoe UI', Arial, sans-serif",
        "title":     "\u0648\u0631\u0642\u0629 \u0627\u0644\u0625\u062c\u0627\u0628\u0629",
        "choices":   ["\u0627", "\u0628", "\u062c", "\u062f", "\u0647"],
        "lbl": {
            "student_name": "\u0627\u0633\u0645 \u0627\u0644\u0637\u0627\u0644\u0628",
            "student_code": "\u0631\u0642\u0645 \u0627\u0644\u0637\u0627\u0644\u0628",
            "exam_code":    "\u0631\u0645\u0632 \u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646",
            "class_name":   "\u0627\u0644\u0641\u0635\u0644",
            "student_qr":   "\u0627\u0644\u0637\u0627\u0644\u0628",
            "exam_qr":      "\u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646",
            "legend":       "\u062f\u0644\u064a\u0644:",
            "empty":        "\u0641\u0627\u0631\u063a",
            "selected":     "\u0627\u0644\u0645\u062e\u062a\u0627\u0631\u0629",
            "hint":         "\u0627\u0645\u0644\u0623 \u0627\u0644\u062f\u0627\u0626\u0631\u0629 \u0643\u0627\u0645\u0644\u0627\u064b \u0628\u0642\u0644\u0645 \u063a\u0627\u0645\u0642",
            "footer":       "\u0648\u0631\u0642\u0629 \u0625\u062c\u0627\u0628\u0629 OMR  |  100 \u0633\u0624\u0627\u0644  |  5 \u062e\u064a\u0627\u0631\u0627\u062a  |  A4",
        },
    },
}

DATA = {
    "student_name": "Ahmed Ali",
    "student_code": "STU-20240001",
    "exam_code":    "EXAM-2024-MATH-01",
    "class_name":   "Grade 10-A",
}

# ── layout parameters auto-calculated from question count ─────────────────────
def layout_for(n: int) -> dict:
    """Return CSS/grid parameters that fill A4 comfortably for any question count."""
    if n <= 20:
        return dict(n_cols=2, bubble=26, qn_size=16, ch_size=14,
                    row_py=8,  header_every=5,  col_gap=28)
    if n <= 50:
        # 4 cols × 13 rows = 52 slots — fits with room to breathe
        return dict(n_cols=4, bubble=20, qn_size=14, ch_size=12,
                    row_py=5,  header_every=5,  col_gap=12)
    # 100
    return     dict(n_cols=4, bubble=18, qn_size=13, ch_size=11,
                    row_py=3,  header_every=5,  col_gap=12)

# ── QR helper ─────────────────────────────────────────────────────────────────
def qr_b64(text: str) -> str:
    qr = qrcode.QRCode(version=None,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=7, border=2)
    qr.add_data(text); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── question grid (header row + question rows) ────────────────────────────────
def build_grid(n: int, choices: list, lp: dict) -> str:
    n_cols       = lp["n_cols"]
    header_every = lp["header_every"]
    rpc = (n + n_cols - 1) // n_cols     # rows per column

    def col_header():
        return (
            '<div class="col-header">'
            '<span class="ch-spacer"></span>'
            + "".join(f'<span class="ch-lbl">{ch}</span>' for ch in choices)
            + '</div>'
        )

    # Bubble-zone bar: [num-spacer (transparent)] [bubble-bar (black)] in DOM order.
    # With RTL direction the flex renders them right→left, so num-spacer lands on
    # the right (above .qn) and bubble-bar lands on the left (above .bs) — matching
    # the question-row layout exactly.  The bar's right edge in the warped image is
    # the precise bubble/number boundary the detector uses for masking.
    def col_anchor_bar():
        return (
            '<div class="col-anchor">'
            '<span class="col-num-spacer"></span>'
            '<span class="col-bubble-bar"></span>'
            '</div>'
        )

    cols = []
    for col in range(n_cols):
        start = col * rpc + 1
        end   = min((col + 1) * rpc, n)
        if start > n:
            break
        rows = []
        actual_q_count = end - start + 1
        for i, q in enumerate(range(start, end + 1)):
            if i % header_every == 0:
                rows.append(col_header())
            circles = "".join('<span class="bubble"></span>' for _ in choices)
            rows.append(
                f'<div class="qr">'
                f'<span class="qn">{q}</span>'
                f'<span class="bs">{circles}</span>'
                f'</div>'
            )
        # pad short columns with invisible placeholder rows so flex heights match
        for i in range(actual_q_count, rpc):
            if i % header_every == 0:
                rows.append('<div class="col-header col-header-phantom"></div>')
            rows.append('<div class="qr qr-phantom"></div>')
        cols.append(
            '<div class="qc">'
            + col_anchor_bar() +
            "".join(rows) +
            col_anchor_bar() +
            '</div>'
        )
    return '<div class="qg">' + "".join(cols) + "</div>"

# ── full HTML ─────────────────────────────────────────────────────────────────
def generate_html(lang_key: str, data: dict, n_questions: int) -> str:
    L  = LANGS[lang_key]
    lb = L["lbl"]
    d  = L["dir"]
    rtl = (d == "rtl")
    ch  = L["choices"]
    lp  = layout_for(n_questions)
    sqr = qr_b64(data["student_code"])
    eqr = qr_b64(data["exam_code"])
    grid = build_grid(n_questions, ch, lp)

    bubble_px  = lp["bubble"]
    qn_size    = lp["qn_size"]
    ch_size    = lp["ch_size"]
    row_py     = lp["row_py"]
    col_gap    = lp["col_gap"]
    sep_border = "border-left" if rtl else "border-right"
    # Separator side for the bubble-set (.bs): faces the number (.qn)
    sep_bs_side  = "right" if rtl else "left"
    sep_num_side = "left"  if rtl else "right"

    # Dynamic footer
    n_choices  = len(ch)
    footer_txt = lb["footer"].replace("100", str(n_questions)).replace("5", str(n_choices))

    return f"""<!DOCTYPE html>
<html lang="{L['lang_attr']}" dir="{d}">
<head>
<meta charset="UTF-8"/>
<title>{L['title']}</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}

  body {{
    font-family: {L['font']};
    background: #aaa;
    color: #111;
    direction: {d};
  }}

  /* ── A4 page ── */
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

  /* ── Corner anchors (large, 4 corners) ── */
  .anchor {{
    position: absolute;
    background: #000;
    z-index: 20;
  }}
  .anchor.corner {{
    width: 12mm; height: 12mm;
  }}
  .anchor.tl {{ top:4mm;    left:4mm;   }}
  .anchor.tr {{ top:4mm;    right:4mm;  }}
  .anchor.bl {{ bottom:4mm; left:4mm;   }}
  .anchor.br {{ bottom:4mm; right:4mm;  }}

  /* ── Mid-edge anchors (2, left+right centre) ── */
  .anchor.mid {{
    width: 7mm; height: 7mm;
    top: 50%;
    transform: translateY(-50%);
  }}
  .anchor.ml {{ left:0; }}
  .anchor.mr {{ right:0; }}

  /* ── Content ── */
  .content {{
    flex: 1;
    padding: 16mm 18mm 8mm;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  /* ── Title ── */
  .title {{
    text-align: center;
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 1.5px;
    border-bottom: 3px double #222;
    padding-bottom: 7px;
    margin-bottom: 2px;
  }}

  /* ── Header ── */
  .header {{
    display: flex;
    gap: 12px;
    align-items: stretch;
  }}

  .meta {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .meta-row {{ display:flex; gap:12px; }}

  .field {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .field.wide {{ flex: 2; }}

  .field label {{
    font-size: 9px;
    font-weight: 800;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  .field .val {{
    border-bottom: 2px solid #222;
    padding: 3px 5px;
    font-size: 14px;
    font-weight: 700;
    min-height: 24px;
    line-height: 1.4;
  }}

  /* ── QR ── */
  .qrs {{ display:flex; gap:10px; align-items:center; }}
  .qrbox {{
    display:flex; flex-direction:column; align-items:center; gap:3px;
  }}
  .qrbox img  {{ width:20mm; height:20mm; display:block; }}
  .qrbox span {{
    font-size:8.5px; color:#444; font-weight:800;
    text-transform:uppercase; letter-spacing:0.4px;
  }}

  /* ── Legend bar ── */
  .legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11px;
    font-weight: 600;
    border-top: 2px solid #aaa;
    border-bottom: 2px solid #aaa;
    padding: 5px 2px;
    background: #f8f8f8;
  }}
  .legend .ltitle {{ font-size:12px; font-weight:900; }}
  .legend-sample {{
    display: flex; align-items:center; gap: 4px;
  }}
  .legend .hint {{
    font-size: 9.5px; color: #555; font-weight:500;
    margin-{"left" if not rtl else "right"}: auto;
  }}

  /* ── Bubble (in legend) ── */
  .bubble-sm {{
    display:inline-block;
    width:16px; height:16px;
    border-radius:50%;
    border:2px solid #111;
    background:#fff;
    vertical-align:middle;
  }}
  .bubble-sm.filled {{ background:#111; }}

  /* ── Question grid ── */
  .qg {{
    flex: 1;
    display: flex;
    gap: 0;
    align-items: stretch;
  }}

  .qc {{
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 0 {col_gap // 2}px;
    {sep_border}: 2px solid #ccc;
  }}
  .qc:last-child {{ {sep_border}: none; }}

  /* ── Column bubble-zone bar (top + bottom of each column) ── */
  /* Mirrors the .qr flex layout so the black bar aligns exactly over .bs  */
  /* and the transparent spacer aligns over .qn.  The bar's far edge from  */
  /* .qn is the precise bubble/number boundary the OMR detector uses.      */
  .col-anchor {{
    display: flex;
    align-items: stretch;
    padding: 2px 0;
  }}
  .col-num-spacer {{
    min-width: 28px;
    flex-shrink: 0;
  }}
  .col-bubble-bar {{
    flex: 1;
    height: 5mm;
    background: #000;
    border-radius: 1px;
  }}

  /* ── Column choice-letter header ── */
  .col-header {{
    display: flex;
    align-items: center;
    padding: 3px 0 2px;
    border-bottom: 1.5px solid #999;
    margin-bottom: 1px;
    background: #f4f4f4;
  }}

  .ch-spacer {{
    min-width: 28px;
    flex-shrink: 0;
  }}

  .ch-lbl {{
    flex: 1;
    text-align: center;
    font-size: {ch_size}px;
    font-weight: 900;
    color: #444;
    letter-spacing: 0.5px;
  }}

  /* ── Single question row ── */
  .qr {{
    flex: 1;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #eee;
    min-height: {bubble_px + 4}px;
  }}
  .qr-phantom {{
    border-bottom: none;
    visibility: hidden;
  }}
  .col-header-phantom {{
    visibility: hidden;
  }}

  .qn {{
    font-size: {qn_size}px;
    font-weight: 900;
    min-width: 28px;
    text-align: {"right" if not rtl else "left"};
    color: #000;
    flex-shrink: 0;
    line-height: 1;
    padding-{"right" if not rtl else "left"}: 3px;
    padding-{sep_num_side}: 6px;
  }}

  .bs {{
    flex: 1;
    display: flex;
    justify-content: space-around;
    align-items: center;
    border-{sep_bs_side}: 3px solid #333;
    padding-{sep_bs_side}: 7px;
  }}

  /* ── Bubble ── */
  .bubble {{
    display: inline-block;
    width:  {bubble_px}px;
    height: {bubble_px}px;
    border-radius: 50%;
    border: 2px solid #111;
    background: #fff;
    flex-shrink: 0;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    font-size: 9px;
    font-weight: 600;
    color: #777;
    border-top: 1px solid #ccc;
    padding: 4px 0 2px;
    letter-spacing: 0.3px;
  }}

  @media print {{
    body {{ background:none; }}
    .sheet {{ margin:0; box-shadow:none; }}
  }}
</style>
</head>
<body>
<div class="sheet">

  <!-- 4 corner anchors -->
  <div class="anchor corner tl"></div>
  <div class="anchor corner tr"></div>
  <div class="anchor corner bl"></div>
  <div class="anchor corner br"></div>

  <!-- 2 mid-edge anchors -->
  <div class="anchor mid ml"></div>
  <div class="anchor mid mr"></div>

  <div class="content">

    <!-- Title -->
    <div class="title">{L['title']}</div>

    <!-- Header -->
    <div class="header">
      <div class="meta">
        <div class="meta-row">
          <div class="field wide">
            <label>{lb['student_name']}</label>
            <div class="val">{data['student_name']}</div>
          </div>
          <div class="field">
            <label>{lb['student_code']}</label>
            <div class="val">{data['student_code']}</div>
          </div>
        </div>
        <div class="meta-row">
          <div class="field">
            <label>{lb['exam_code']}</label>
            <div class="val">{data['exam_code']}</div>
          </div>
          <div class="field">
            <label>{lb['class_name']}</label>
            <div class="val">{data['class_name']}</div>
          </div>
        </div>
      </div>
      <div class="qrs">
        <div class="qrbox">
          <img src="data:image/png;base64,{sqr}" alt="Student QR"/>
          <span>{lb['student_qr']}</span>
        </div>
        <div class="qrbox">
          <img src="data:image/png;base64,{eqr}" alt="Exam QR"/>
          <span>{lb['exam_qr']}</span>
        </div>
      </div>
    </div>

    <!-- Legend -->
    <div class="legend">
      <span class="ltitle">{lb['legend']}</span>
      <span class="legend-sample">
        <span class="bubble-sm"></span>&nbsp;{lb['empty']}
      </span>
      &nbsp;&nbsp;
      <span class="legend-sample">
        <span class="bubble-sm filled"></span>&nbsp;{lb['selected']}
      </span>
      <span class="hint">{lb['hint']}</span>
    </div>

    <!-- Grid -->
    {grid}

    <!-- Footer -->
    <div class="footer">{footer_txt}</div>

  </div>
</div>
</body>
</html>
"""

# ── PDF ───────────────────────────────────────────────────────────────────────
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
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    sizes = [20, 50, 100]
    for n in sizes:
        for lang in ("en", "ar"):
            stem = f"answer_sheet_v3_{n}q_{lang}"
            html = generate_html(lang, DATA, n)
            pathlib.Path(f"{stem}.html").write_text(html, encoding="utf-8")
            print(f"[{lang.upper()} {n:>3}Q] -> {stem}.html")
            if args.pdf:
                to_pdf(f"{stem}.html", f"{stem}.pdf")
