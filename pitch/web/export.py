"""Export the web deck to PDF (USB backup for the jury) and, optionally, one PNG per slide.

Uses the system Google Chrome through Playwright, so nothing is downloaded except the
Playwright Python package:

    uv run --no-project --with playwright python pitch/web/export.py
    uv run --no-project --with playwright python pitch/web/export.py --png /tmp/slides
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
CHROME = "/usr/bin/google-chrome-stable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=HERE / "EdgeSense-pitch.pdf")
    ap.add_argument("--png", type=Path, help="also write slide-NN.png (1920x1080) into this folder")
    ap.add_argument("--presenter", action="store_true", help="with --png, also capture the presenter view")
    args = ap.parse_args()
    url = (HERE / "index.html").as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)

        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url)
        page.evaluate("document.fonts.ready")
        page.pdf(path=str(args.pdf), prefer_css_page_size=True, print_background=True)
        print("pdf:", args.pdf)

        if args.png:
            args.png.mkdir(parents=True, exist_ok=True)
            # reduced motion = every element in its final, settled state
            shot = browser.new_page(viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
            shot.goto(url + "#1")
            shot.evaluate("document.fonts.ready")
            n = shot.evaluate("document.querySelectorAll('#deck > .slide').length")
            for i in range(1, n + 1):
                shot.evaluate(f"location.hash = '#{i}'")
                shot.wait_for_timeout(500)
                out = args.png / f"slide-{i:02d}.png"
                shot.screenshot(path=str(out))
                print("png:", out)
            if args.presenter:
                pv = browser.new_page(viewport={"width": 1400, "height": 860}, reduced_motion="reduce")
                pv.goto(url + "#presenter")
                pv.evaluate("document.fonts.ready")
                pv.keyboard.press("ArrowRight")
                pv.keyboard.press("ArrowRight")
                pv.wait_for_timeout(1500)
                pv.screenshot(path=str(args.png / "presenter.png"))
                print("png:", args.png / "presenter.png")
        browser.close()


if __name__ == "__main__":
    main()
