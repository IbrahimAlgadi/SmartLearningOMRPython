"""
OMR Answer Sheet generator — English & Arabic.
Outputs self-contained HTML + PDF (Playwright/Chromium).
"""

import qrcode, base64, io, random, pathlib

# ─── LANGUAGE DEFINITIONS ────────────────────────────────────────────────────

LANGUAGES = {
    "en": {
        "lang_attr":  "en",
        "dir":        "ltr",
        "font":       "'Segoe UI', Arial, sans-serif",
        "title_default": "Answer Page",
        "lbl": {
            "student_name": "Student Name",
            "class_name":   "Class",
            "grade":        "Grade",
            "student_code": "Student Code",
            "exam_code":    "Exam Code",
            "legend":       "Legend:",
            "empty":        "Not answered",
            "filled":       "Selected",
            "qr_student":   "Student",
            "qr_exam":      "Exam",
            "footer":       "OMR Answer Sheet &nbsp;|&nbsp; 100 Questions &nbsp;|&nbsp; A4",
        },
        "choices": {
            "AB":   ["A", "B"],
            "ABC":  ["A", "B", "C"],
            "ABCD": ["A", "B", "C", "D"],
        },
        "type_names": {"AB": "AB", "ABC": "ABC", "ABCD": "ABCD"},
        "legend_first": "A",  # letter shown in legend bubble
    },
    "ar": {
        "lang_attr":  "ar",
        "dir":        "rtl",
        "font":       "'Segoe UI', 'Noto Naskh Arabic', 'Arial', sans-serif",
        "title_default": "ورقة الإجابة",
        "lbl": {
            "student_name": "اسم الطالب",
            "class_name":   "الفصل",
            "grade":        "الصف",
            "student_code": "رقم الطالب",
            "exam_code":    "رمز الامتحان",
            "legend":       "دليل:",
            "empty":        "لم يُجب",
            "filled":       "الإجابة المختارة",
            "qr_student":   "الطالب",
            "qr_exam":      "الامتحان",
            "footer":       "ورقة إجابة OMR &nbsp;|&nbsp; 100 سؤال &nbsp;|&nbsp; A4",
        },
        "choices": {
            "AB":   ["أ", "ب"],
            "ABC":  ["أ", "ب", "ج"],
            "ABCD": ["أ", "ب", "ج", "د"],
        },
        "type_names": {"AB": "أ-ب", "ABC": "أ-ب-ج", "ABCD": "أ-ب-ج-د"},
        "legend_first": "أ",
    },
}

# ─── RANDOM OVERRIDES ────────────────────────────────────────────────────────

def make_overrides(n_total=100, n_ab=10, n_abc=15, seed=None):
    rng = random.Random(seed)
    qs  = list(range(1, n_total + 1))
    rng.shuffle(qs)
    out = {}
    for q in qs[:n_ab]:             out[q] = "AB"
    for q in qs[n_ab:n_ab + n_abc]: out[q] = "ABC"
    return out

# Shared overrides so both language sheets are identical in structure
OVERRIDES = make_overrides()

# ─── CONFIGS ─────────────────────────────────────────────────────────────────

CONFIG_EN = {
    "title":        "Answer Page",
    "class_name":   "Grade 10-A",
    "grade":        "10",
    "student_name": "Ahmed Ali",
    "student_code": "STU-20240001",
    "exam_code":    "EXAM-2024-MATH-01",
    "default_type": "ABCD",
    "overrides":    OVERRIDES,
    "lang":         "en",
}

CONFIG_AR = {
    "title":        "ورقة الإجابة",
    "class_name":   "10-أ",
    "grade":        "10",
    "student_name": "أحمد علي",
    "student_code": "STU-20240001",
    "exam_code":    "EXAM-2024-MATH-01",
    "default_type": "ABCD",
    "overrides":    OVERRIDES,
    "lang":         "ar",
}

# ─── QR HELPER ───────────────────────────────────────────────────────────────

def qr_base64(data: str, box_size: int = 6, border: int = 2) -> str:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─── QUESTION GRID ───────────────────────────────────────────────────────────

def build_questions(n: int, default_type: str, overrides: dict, choices: dict) -> str:
    all_labels = choices.get("ABCD", list(choices.values())[-1])   # full 4-choice set
    rows = []
    for q in range(1, n + 1):
        qtype       = overrides.get(q, default_type)
        active_set  = set(choices.get(qtype, all_labels))
        bubbles = "".join(
            f'<span class="bubble">{l}</span>'
            for l in all_labels if l in active_set
        )
        rows.append(
            f'<div class="q-row">'
            f'<span class="q-num">{q}.</span>'
            f'<span class="bubbles">{bubbles}</span>'
            f'</div>'
        )
    col_size = (n + 3) // 4
    cols = [
        '<div class="q-col">' + "".join(rows[c*col_size : min((c+1)*col_size, n)]) + "</div>"
        for c in range(4)
    ]
    return '<div class="q-grid">' + "".join(cols) + "</div>"


def type_summary(overrides: dict, default_type: str, type_names: dict, n: int = 100) -> str:
    from collections import Counter
    counts = Counter(overrides.get(q, default_type) for q in range(1, n + 1))
    order  = ["AB", "ABC", "ABCD"]
    parts  = [f"{counts[t]}&times;{type_names[t]}" for t in order if counts.get(t)]
    return " &nbsp;|&nbsp; ".join(parts)

# ─── HTML GENERATOR ──────────────────────────────────────────────────────────

def generate_html(cfg: dict, output_path: str = "answer_page.html") -> str:
    L          = LANGUAGES[cfg.get("lang", "en")]
    lbl        = L["lbl"]
    overrides  = cfg.get("overrides", {})
    student_qr = qr_base64(cfg["student_code"])
    exam_qr    = qr_base64(cfg["exam_code"])
    questions  = build_questions(100, cfg["default_type"], overrides, L["choices"])
    summary    = type_summary(overrides, cfg["default_type"], L["type_names"])
    lf         = L["legend_first"]
    d          = L["dir"]
    # For RTL: push the summary badge to the left (start of row)
    summary_margin = "margin-right:auto;" if d == "rtl" else "margin-left:auto;"

    html = f"""<!DOCTYPE html>
<html lang="{L['lang_attr']}" dir="{d}">
<head>
<meta charset="UTF-8"/>
<title>{cfg['title']}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: {L['font']};
    font-size: 11px;
    background: #ccc;
    color: #111;
    direction: {d};
  }}

  .sheet {{
    position: relative;
    width: 210mm;
    height: 297mm;
    overflow: hidden;
    margin: 0 auto;
    background: #fff;
    display: flex;
    flex-direction: column;
  }}

  /* ── Corner anchors (OMR fiducial markers) ── */
  .anchor {{
    position: absolute;
    width: 12mm;
    height: 12mm;
    background: #000;
    z-index: 10;
  }}
  .anchor.tl {{ top: 5mm;    left: 5mm;  }}
  .anchor.tr {{ top: 5mm;    right: 5mm; }}
  .anchor.bl {{ bottom: 5mm; left: 5mm;  }}
  .anchor.br {{ bottom: 5mm; right: 5mm; }}

  .content {{
    flex: 1;
    padding: 14mm 22mm;
    display: flex;
    flex-direction: column;
  }}

  /* ── Title ── */
  .page-title {{
    text-align: center;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 1px;
    border-bottom: 3px double #222;
    padding-bottom: 5px;
    margin-bottom: 8px;
  }}

  /* ── Header row: meta fields + QR ── */
  .header-row {{
    display: flex;
    gap: 12px;
    align-items: stretch;
    margin-bottom: 8px;
  }}

  .meta-fields {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    justify-content: space-between;
  }}

  .meta-line {{
    display: flex;
    gap: 10px;
  }}

  .meta-field {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }}

  .meta-field label {{
    font-size: 8px;
    letter-spacing: 0.5px;
    color: #666;
    font-weight: 700;
  }}

  .meta-field .field-value {{
    border-bottom: 1.5px solid #333;
    padding: 1px 4px;
    font-size: 12px;
    font-weight: 600;
    min-height: 18px;
  }}

  /* ── QR codes ── */
  .qr-section {{
    display: flex;
    gap: 10px;
    align-items: center;
    border: 1px solid #bbb;
    border-radius: 5px;
    padding: 6px 8px;
    background: #f8f8f8;
    flex-shrink: 0;
  }}

  .qr-block {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
  }}

  .qr-block img {{
    width: 72px;
    height: 72px;
    display: block;
    border: 1px solid #ddd;
  }}

  .qr-label {{
    font-size: 7px;
    letter-spacing: 0.5px;
    color: #555;
    font-weight: 700;
  }}

  .qr-code-text {{
    font-size: 6.5px;
    color: #888;
    font-family: monospace;
    direction: ltr;
  }}

  /* ── Separator ── */
  .section-sep {{
    border: none;
    border-top: 2px solid #222;
    margin: 6px 0;
  }}

  /* ── Legend ── */
  .legend {{
    display: flex;
    gap: 14px;
    align-items: center;
    margin-bottom: 6px;
    font-size: 9px;
    color: #444;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  .legend-bubble {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 2px solid #222;
    font-size: 8px;
    font-weight: 700;
    flex-shrink: 0;
  }}

  .legend-bubble.filled {{
    background: #222;
    color: #fff;
  }}

  /* ── Question grid ── */
  .q-grid {{
    display: flex;
    gap: 6px;
    flex: 1;
  }}

  .q-col {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0px;
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 4px 3px;
    background: #fdfdfd;
  }}

  .q-row {{
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 2px 0;
    border-bottom: 1px solid #ddd;
  }}

  .q-row:last-child {{ border-bottom: none; }}

  .q-num {{
    font-size: 8px;
    font-weight: 700;
    color: #555;
    width: 20px;
    text-align: center;
    flex-shrink: 0;
    font-family: monospace;
    direction: ltr;
  }}

  .bubbles {{
    display: flex;
    gap: 0px;
    flex-wrap: nowrap;
  }}

  .bubble {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2px solid #555;
    font-size: 11px;
    font-weight: 400;
    color: #999;
    cursor: default;
    user-select: none;
    flex-shrink: 0;
    margin: 0 1px;
  }}

  .bubble.inactive {{
    border-color: #d0d0d0;
    color: #d0d0d0;
    pointer-events: none;
  }}

  /* ── Footer ── */
  .footer {{
    margin-top: 8px;
    text-align: center;
    font-size: 7.5px;
    color: #bbb;
    border-top: 1px solid #e0e0e0;
    padding-top: 4px;
  }}

  @media print {{
    body {{ background: #fff; }}
    .sheet {{ margin: 0; }}
    @page {{ size: A4; margin: 0; }}
  }}

  @media screen {{
    .bubble:hover {{ background: #333; color: #fff; }}
    .sheet {{ box-shadow: 0 4px 24px rgba(0,0,0,.25); margin: 10mm auto; }}
  }}
</style>
</head>
<body>

<div class="sheet">

  <!-- OMR corner anchors — do not remove -->
  <div class="anchor tl"></div>
  <div class="anchor tr"></div>
  <div class="anchor bl"></div>
  <div class="anchor br"></div>

  <div class="content">

    <div class="page-title">{cfg['title']}</div>

    <div class="header-row">
      <div class="meta-fields">
        <div class="meta-line">
          <div class="meta-field">
            <label>{lbl['student_name']}</label>
            <div class="field-value">{cfg['student_name']}</div>
          </div>
          <div class="meta-field">
            <label>{lbl['student_code']}</label>
            <div class="field-value" style="direction:ltr;">{cfg['student_code']}</div>
          </div>
        </div>
        <div class="meta-line">
          <div class="meta-field">
            <label>{lbl['class_name']}</label>
            <div class="field-value">{cfg['class_name']}</div>
          </div>
          <div class="meta-field">
            <label>{lbl['grade']}</label>
            <div class="field-value">{cfg['grade']}</div>
          </div>
          <div class="meta-field">
            <label>{lbl['exam_code']}</label>
            <div class="field-value" style="direction:ltr;">{cfg['exam_code']}</div>
          </div>
        </div>
      </div>

      <div class="qr-section">
        <div class="qr-block">
          <img src="data:image/png;base64,{student_qr}" alt="Student QR"/>
          <span class="qr-label">{lbl['qr_student']}</span>
          <span class="qr-code-text">{cfg['student_code']}</span>
        </div>
        <div class="qr-block">
          <img src="data:image/png;base64,{exam_qr}" alt="Exam QR"/>
          <span class="qr-label">{lbl['qr_exam']}</span>
          <span class="qr-code-text">{cfg['exam_code']}</span>
        </div>
      </div>
    </div>

    <hr class="section-sep"/>

    <div class="legend">
      <strong>{lbl['legend']}</strong>
      <div class="legend-item">
        <span class="legend-bubble">{lf}</span> {lbl['empty']}
      </div>
      <div class="legend-item">
        <span class="legend-bubble filled">{lf}</span> {lbl['filled']}
      </div>
      <div class="legend-item" style="{summary_margin} color:#888; font-size:8.5px;">
        {summary}
      </div>
    </div>

    {questions}

    <div class="footer">{lbl['footer']}</div>

  </div>
</div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML -> {output_path}")
    return output_path

# ─── HTML → PDF ──────────────────────────────────────────────────────────────

def html_to_pdf(html_path: str, pdf_path: str = None) -> str:
    from playwright.sync_api import sync_playwright
    if pdf_path is None:
        pdf_path = str(pathlib.Path(html_path).with_suffix(".pdf"))
    uri = pathlib.Path(html_path).resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(uri, wait_until="networkidle")
        page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()
    print(f"[OK] PDF  -> {pdf_path}")
    return pdf_path

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for cfg, name in [(CONFIG_EN, "answer_page_en"), (CONFIG_AR, "answer_page_ar")]:
        html = generate_html(cfg, f"{name}.html")
        html_to_pdf(html, f"{name}.pdf")
