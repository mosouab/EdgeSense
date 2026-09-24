"""Compile SCRIPT.md into a printable A4 PDF, with a thumbnail of each slide next to its text.

Uses the system Google Chrome through Playwright (same as export.py):

    uv run --no-project --with playwright --with markdown python pitch/web/script_pdf.py
"""

import argparse
import re
import tempfile
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
CHROME = "/usr/bin/google-chrome-stable"

CSS = """
@font-face { font-family: "Archivo"; src: url("fonts/archivo-latin-wght-normal.woff2") format("woff2-variations"); font-weight: 100 900; }
@font-face { font-family: "Archivo"; src: url("fonts/archivo-latin-ext-wght-normal.woff2") format("woff2-variations"); font-weight: 100 900;
  unicode-range: U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+1E00-1E9F, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF; }
@font-face { font-family: "Plex Mono"; src: url("fonts/ibm-plex-mono-latin-400-normal.woff2") format("woff2"); font-weight: 400; }
@font-face { font-family: "Plex Mono"; src: url("fonts/ibm-plex-mono-latin-600-normal.woff2") format("woff2"); font-weight: 600; }
@page { size: A4; margin: 16mm 16mm 18mm; }
:root { --navy: #0a3d68; --blue: #005291; --ink: #0e2e52; --ink-2: #4a6480; --muted: #7d91a6; --line: #d3dde8; --mist: #f1f5f9;
  --amber: #e2b818; --orange: #d47d16; --orange-ink: #a4580b;
  --stripe: linear-gradient(90deg, #0c4068, #4a95ac 30%, #a2b566 50%, #e2b818 68%, #d47d16); }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 400 10.5pt/1.5 "Archivo", sans-serif; color: var(--ink); }
.mast { border-bottom: 0; padding-bottom: 5mm; margin-bottom: 5mm; position: relative; }
.mast::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2.2mm; background: var(--stripe); }
.kicker { font: 600 7.5pt "Plex Mono", monospace; letter-spacing: .14em; text-transform: uppercase; color: #2f7f98; margin: 0 0 1.5mm; }
h1 { font-size: 24pt; line-height: 1.1; font-weight: 800; letter-spacing: -.02em; color: var(--blue); margin: 0; }
.sub { margin: 1.5mm 0 0; color: var(--ink-2); font-size: 10.5pt; }
.intro p { margin: 0 0 2mm; color: var(--ink-2); }
code { font: 600 9pt "Plex Mono", monospace; background: var(--mist); padding: 0 1.2mm; border-radius: 1mm; color: var(--ink); }
.slide { display: grid; grid-template-columns: 58mm 1fr; gap: 6mm; padding: 5mm 0; border-top: .3mm solid var(--line); break-inside: avoid; }
.slide:first-of-type { border-top: 0; }
.thumb img { width: 58mm; display: block; border-radius: 1.2mm; box-shadow: 0 0 0 .25mm var(--line); }
.thumb .time { margin-top: 2mm; font: 600 8pt "Plex Mono", monospace; letter-spacing: .06em; color: var(--orange-ink); }
.thumb .dur { font: 400 7.5pt "Plex Mono", monospace; color: var(--muted); }
h2 { margin: 0 0 2mm; font-size: 13.5pt; line-height: 1.2; font-weight: 750; color: var(--blue); }
h2 .n { display: inline-grid; place-items: center; width: 6.2mm; height: 6.2mm; margin-right: 2mm; border-radius: 50%; background: var(--blue); color: #fff; font-size: 9pt; vertical-align: .4mm; }
.text blockquote { margin: 0 0 2.2mm; padding: 0; border: 0; }
.text blockquote p { margin: 0 0 2mm; font-size: 12pt; line-height: 1.5; }
.text > p { margin: 0 0 2mm; }
strong { color: var(--blue); font-weight: 700; }
.cue { display: block; font: 400 8.5pt/1.4 "Plex Mono", monospace; color: var(--muted); margin: 1mm 0; }
.cue::before { content: "▸ "; color: var(--orange); }
.cue--inline { display: inline; margin: 0; }
.pause { color: var(--orange); font-weight: 700; padding: 0 .6mm; }
.pause--long { color: var(--orange); font-weight: 800; letter-spacing: -.05em; padding: 0 .8mm; }
.fill { background: #fff3c4; box-shadow: 0 0 0 .3mm var(--amber); border-radius: .8mm; padding: 0 .8mm; }
.todo { display: inline; background: #fff3c4; color: var(--orange-ink); font: 600 8.5pt "Plex Mono", monospace; padding: .2mm 1mm; border-radius: .8mm; }
.part { break-before: page; }
.part.keep { break-before: auto; margin-top: 7mm; }
.part h2 { font-size: 15pt; margin: 0 0 3mm; padding-bottom: 2mm; border-bottom: .8mm solid var(--blue); }
.part ol { margin: 0; padding-left: 6mm; }
.part ol li { margin: 0 0 1.6mm; font-size: 11pt; }
.qa p { margin: 0 0 1.4mm; break-inside: avoid; }
.qa p.q { margin-top: 3.6mm; break-after: avoid; }
.qa p.q strong { font-size: 11pt; }
.qa > p:first-of-type { margin-top: 0; color: var(--ink-2); }
.pitch blockquote { margin: 0; padding: 4mm 5mm; background: var(--mist); border-left: 1.4mm solid var(--blue); border-radius: 1mm; font-size: 11.5pt; }
.pitch blockquote p { margin: 0; }
"""

FOOTER = """<div style="width:100%; padding: 0 16mm; font: 7.5pt monospace; color: #7d91a6; display:flex; justify-content:space-between;">
<span>EdgeSense · script du pitch (3 min)</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>"""


def decorate(html, spoken):
    """Style stage directions, placeholders, to-dos and (in spoken text only) pause marks."""
    html = html.replace("[Prénom]", "\x00PRENOM\x00")      # keep the placeholder out of the cue rules
    html = re.sub(r"\[(À (?:adapter|compléter)[^\]]*)\]", r'<span class="todo">\1</span>', html)
    # a paragraph made only of a stage direction becomes a block cue
    html = re.sub(r"<p>\[([^\]]+)\]</p>", r'<p class="cue">\1</p>', html)
    html = re.sub(r"(<blockquote>\s*<p>|<br />\s*|<p>)\[([^\]]+)\]\s*(<br />)?", r'\1<span class="cue">\2</span>', html)
    html = re.sub(r"\[([^\]]+)\]", r'<span class="cue cue--inline">\1</span>', html)
    if spoken:
        html = re.sub(r" //(?=\s|<)", ' <span class="pause--long">//</span>', html)
        html = re.sub(r" /(?=\s|<)", ' <span class="pause">/</span>', html)
    html = html.replace("\x00PRENOM\x00", '<span class="fill">[Prénom]</span>')
    return french(html)


def french(html):
    """French typography in text nodes: curly apostrophes, no-break spaces before : ; ? ! and inside « »."""
    def fix(m):
        t = m.group(1).replace("'", "\u2019")
        t = re.sub(r" ([:;?!»])", "\u202f\\1", t).replace("« ", "«\u202f")
        return ">" + t + "<"
    return re.sub(r">([^<]+)<", fix, html)


def md(text):
    return markdown.markdown(text, extensions=["nl2br"])


def build_html(script, thumbs, durations):
    parts = script.split("\n---\n")
    intro_md, slides_md, rest = parts[0], parts[1], parts[2:]
    title = re.search(r"^# (.+)$", intro_md, re.M).group(1)
    intro_body = re.sub(r"^# .+\n", "", intro_md, flags=re.M).strip()

    out = [f'<header class="mast"><p class="kicker">Think It &amp; Do It · Université d’Évry</p><h1>{title}</h1>'
           f'<p class="sub">Même texte que la vue présentateur du diaporama (touche P).</p></header>',
           f'<section class="intro">{decorate(md(intro_body), spoken=False)}</section>']

    sections = re.split(r"\n## ", "\n" + slides_md.strip())[1:]
    for i, sec in enumerate(sections):
        head, body = sec.split("\n", 1)
        m = re.match(r"(\d+) · (.+?) \((\d+:\d\d) → (\d+:\d\d)\)", head)
        n, name, t0, t1 = m.groups()
        out.append(
            f'<section class="slide"><div class="thumb"><img src="{thumbs[i].as_uri()}" alt="Diapo {n}">'
            f'<div class="time">{t0} → {t1}</div><div class="dur">{durations[i]} s</div></div>'
            f'<div class="text"><h2><span class="n">{n}</span>{french(">" + name + "<")[1:-1]}</h2>{decorate(md(body), spoken=True)}</div></section>')

    for j, part in enumerate(rest):
        head, body = part.strip().split("\n", 1)
        body_html = decorate(md(body), spoken=False)
        cls = "part"
        if "Questions" in head:
            body_html = re.sub(r"<p><strong>(«.+?»)</strong>", r'<p class="q"><strong>\1</strong>', body_html)
            cls += " qa"
        elif "Phrase-pitch" in head:
            cls += " pitch keep"
        elif "structure" in head:
            cls += " keep"
        out.append(f'<section class="{cls}"><h2>{french(">" + head.lstrip("# ") + "<")[1:-1]}</h2>{body_html}</section>')

    return f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{"".join(out)}</body></html>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "EdgeSense-script.pdf")
    args = ap.parse_args()
    script = (HERE / "SCRIPT.md").read_text()
    deck_url = (HERE / "index.html").as_uri()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        tmp = Path(tmp)
        browser = p.chromium.launch(executable_path=CHROME, headless=True)

        # slide thumbnails, settled state (reduced motion), half resolution
        shot = browser.new_page(viewport={"width": 960, "height": 540}, reduced_motion="reduce")
        shot.goto(deck_url + "#1")
        shot.evaluate("document.fonts.ready")
        durations = shot.evaluate("[...document.querySelectorAll('#deck > .slide')].map(s => +s.dataset.duration)")
        thumbs = []
        for i in range(1, len(durations) + 1):
            shot.evaluate(f"location.hash = '#{i}'")
            shot.wait_for_timeout(400)
            path = tmp / f"slide-{i:02d}.jpg"
            shot.screenshot(path=str(path), type="jpeg", quality=82)
            thumbs.append(path)

        html_path = HERE / ".script-print.html"   # next to fonts/ so relative font URLs resolve
        html_path.write_text(build_html(script, thumbs, durations))
        try:
            page = browser.new_page()
            page.goto(html_path.as_uri())
            page.evaluate("document.fonts.ready")
            page.pdf(path=str(args.out), format="A4", print_background=True, display_header_footer=True,
                     header_template="<span></span>", footer_template=FOOTER,
                     margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        finally:
            html_path.unlink(missing_ok=True)
        browser.close()
    print("pdf:", args.out)


if __name__ == "__main__":
    main()
