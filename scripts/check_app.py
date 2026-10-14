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
        for mode in ['light', 'dark']:
            for name, w, h in [('desktop', 1366, 900), ('mobile', 390, 844)]:
                pg = await b.new_page(viewport={'width': w, 'height': h}, color_scheme=mode)
                pg.on('pageerror', lambda e: errs.append(str(e)))
                pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
                await pg.goto(URL, wait_until='networkidle'); await pg.wait_for_timeout(1500)
                async def shot(sel, fn, dy=-200):
                    await pg.locator(sel).scroll_into_view_if_needed(); await pg.evaluate(f'window.scrollBy(0,{dy})')
                    await pg.wait_for_timeout(600); await pg.screenshot(path=os.path.join(SHOTS, f'{mode}_{name}_{fn}.png'))
                await pg.screenshot(path=os.path.join(SHOTS, f'{mode}_{name}_01_intro.png'))
                await shot('#countryDonut', '02_country_donut')
                await shot('#modelDonut', '03_family_donut')
                await shot('#scatter', '04_price_explorer', -150)
                await pg.check('#fGold'); await shot('#pitfall', '05_gold_pitfall_filtered', -120); await pg.uncheck('#fGold')
                await pg.click('button[data-p=s2gold]'); await shot('#waterfall', '06_shap_explorer', -300)
                await pg.close()
        # toggle test: light -> dark re-themes charts without errors
        pg = await b.new_page(viewport={'width': 1366, 'height': 900}, color_scheme='light')
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto(URL, wait_until='networkidle'); await pg.wait_for_timeout(1000)
        await pg.click('#themeToggle'); await pg.wait_for_timeout(800)
        t = await pg.evaluate("[document.documentElement.dataset.theme, localStorage.getItem('theme'), document.getElementById('scatter')._fullLayout.font.color]")
        print('toggle ->', t)
        if t[0] != 'dark' or t[1] != 'dark' or t[2].lower() != '#e8e8e8':
            errs.append('theme toggle failed: ' + str(t))
        await b.close()
    print('JS errors:', errs or 'none')
    sys.exit(1 if errs else 0)
asyncio.run(main())
