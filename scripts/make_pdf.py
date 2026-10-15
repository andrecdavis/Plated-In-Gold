"""Build the employer-facing PDF report from README.md.

    python scripts/make_pdf.py [--out reports/Plated_in_Gold_Andre_Davis.pdf]

Markdown -> HTML with pandoc (falls back to the Python `markdown` package if
pandoc is not installed), then HTML -> PDF with headless Chrome/Chromium.
Set CHROME=/path/to/chrome to choose the browser binary.
"""
import argparse, html as htmllib, os, platform, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLE = 'Plated in Gold: What Drives Apple Watch Resale Prices?'
AUTHOR = 'André Davis'
REPO = 'https://github.com/andrecdavis/Plated-In-Gold'
APP = 'https://andrecdavis.github.io/Plated-In-Gold/'
PAGE_BREAKS = ['The UK effect', 'The gold-plating pitfall', 'Modeling: CatBoost']
APPENDIX = '''

## Appendix: project structure and reproduction

Python 3.13; dependencies are pinned in `requirements.txt`. Running `make all` regenerates the figures, executes the analysis notebook and rebuilds the interactive app's data; `make pdf` rebuilds this report.

- `data/raw/` – raw scraped listings (CSV)
- `src/` – cleaning, feature extraction, CatBoost + SHAP modeling, summary statistics, visualization
- `notebooks/` – end-to-end analysis notebook with outputs
- `images/` – figures generated from code
- `app/` – interactive web app, published with GitHub Pages
'''


def find_chrome():
    if os.environ.get('CHROME'):
        if Path(os.environ['CHROME']).exists():
            return os.environ['CHROME']
        sys.exit(f"CHROME is set but not found: {os.environ['CHROME']}")
    cands = []
    if platform.system() == 'Darwin':
        cands += ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                  '/Applications/Chromium.app/Contents/MacOS/Chromium',
                  os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')]
    cands += [shutil.which(n) for n in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome')]
    for c in cands:
        if c and Path(c).exists():
            return c
    sys.exit('Chrome/Chromium not found. Install Google Chrome or set CHROME=/path/to/chrome.')


def report_markdown():
    md = (ROOT / 'README.md').read_text(encoding='utf-8')
    # GitHub-only blocks (cover image, etc.) must not appear in the PDF
    md = re.sub(r'<!-- pdf-exclude-start -->.*?<!-- pdf-exclude-end -->\n?', '', md, flags=re.S)
    md = '## Introduction and question' + md.split('\n## Introduction and question', 1)[1]
    md = re.sub(r'\n## Contents\n.*?(?=\n## )', '\n', md, flags=re.S)
    md = md.split('\n## Project structure and reproduction')[0]
    md = re.sub(r'^((?!- |\d\. ).+)\n(?=- |1\. )', r'\1\n\n', md, flags=re.M)  # lists need a blank line before
    md = re.sub(r'\[([^\]]+)\]\(#[^)]+\)', r'\1', md)  # in-page README links have no target in the PDF
    return md + APPENDIX


def to_html(md):
    if shutil.which('pandoc'):
        return subprocess.run(['pandoc', '-f', 'gfm-tex_math_dollars-tex_math_gfm', '-t', 'html5'], input=md, text=True,
                              capture_output=True, check=True).stdout
    import markdown  # fallback: pip install markdown
    return markdown.markdown(md, extensions=['tables', 'fenced_code', 'md_in_html'])


def build_html(body):
    body = body.replace('src="images/', f'src="{(ROOT / "images").as_uri()}/')
    body = re.sub(r'<p align="center">', '<p class="fig">', body)
    body = body.replace('<p class="fig"><img src="' + (ROOT / 'images').as_uri() + '/01', '<p class="fig pair"><img src="' + (ROOT / 'images').as_uri() + '/01')
    for h in PAGE_BREAKS:
        body = re.sub(rf'<(h[23])( id="[^"]*")?>{re.escape(h)}</h[23]>', rf'<\1\2 class="newpage">{h}</\1>', body)
    css = (ROOT / 'scripts' / 'pdf_style.css').read_text(encoding='utf-8')
    e = htmllib.escape
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(TITLE)}</title><style>{css}</style></head><body>
<header><h1>{e(TITLE)}</h1><p class="author">{e(AUTHOR)}</p>
<p class="links">Code: <a href="{REPO}">{REPO.split("//")[1]}</a> &nbsp;·&nbsp; Interactive app: <a href="{APP}">{APP.split("//")[1].rstrip("/")}</a></p></header>
{body}</body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'reports' / 'Plated_in_Gold_Andre_Davis.pdf'))
    ap.add_argument('--keep-html', action='store_true', help='also write the intermediate HTML next to the PDF')
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(to_html(report_markdown()))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'report.html'; src.write_text(html, encoding='utf-8')
        if a.keep_html:
            out.with_suffix('.html').write_text(html, encoding='utf-8')
        subprocess.run([find_chrome(), '--headless', '--disable-gpu', '--no-sandbox', '--allow-file-access-from-files',
                        '--no-pdf-header-footer', '--run-all-compositor-stages-before-draw', '--virtual-time-budget=10000',
                        f'--print-to-pdf={out}', src.as_uri()], check=True, capture_output=True)
    print(f'wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({out.stat().st_size / 1e6:.2f} MB)')
    app_pdf = ROOT / 'app' / 'report.pdf'
    shutil.copy2(out, app_pdf)
    print(f'copied to {app_pdf.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
