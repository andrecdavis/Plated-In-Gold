"""Serve app/ and load it in headless Chromium; fail on any JS error; refresh app/screenshots/."""
import asyncio, functools, http.server, os, sys, threading
from playwright.async_api import async_playwright
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=APP))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f'http://127.0.0.1:{srv.server_port}/index.html'
SHOTS = os.path.join(APP, 'screenshots'); os.makedirs(SHOTS, exist_ok=True)

async def main():
    errs = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, w, h in [('desktop', 1366, 900), ('mobile', 390, 844)]:
            pg = await b.new_page(viewport={'width': w, 'height': h})
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
            await pg.goto(URL, wait_until='networkidle'); await pg.wait_for_timeout(1500)
            async def shot(sel, fn, dy=-120):
                await pg.locator(sel).scroll_into_view_if_needed(); await pg.evaluate(f'window.scrollBy(0,{dy})')
                await pg.wait_for_timeout(600); await pg.screenshot(path=os.path.join(SHOTS, fn))
            if name == 'desktop':
                await pg.screenshot(path=os.path.join(SHOTS, '01_intro.png'))
                await shot('#glance', '02_data_at_a_glance.png')
                await pg.check('#fGold'); await shot('#pitfall', '03_gold_pitfall_filtered.png')
                await pg.uncheck('#fGold')
                await pg.click('button[data-p=s2gold]'); await shot('#waterfall', '04_shap_listing_explorer.png', -300)
            else:
                await shot('#glance', '05_mobile_data_at_glance.png')
        await b.close()
    print('JS errors:', errs or 'none')
    sys.exit(1 if errs else 0)
asyncio.run(main())
