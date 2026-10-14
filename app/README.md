# Plated in Gold: interactive app

Static site (`index.html`, `style.css`, `app.js`, `data.js`) with Plotly.js from a CDN. Live version: https://andrecdavis.github.io/Plated-In-Gold/

- Run it locally: `cd app && python -m http.server 8000`, then open http://localhost:8000. You can also open `index.html` directly (needs internet for the CDN).
- Rebuild the data: `python app/build_data.py` from the repo root. This writes `data.js` from `src.story.compute_story()`.
- Smoke test: `python scripts/check_app.py` loads the app in headless Chromium, fails on any JS error and refreshes `screenshots/`.
- `AUDIT.md` traces every claim in the app to its computed source.
