PY ?= python

.PHONY: all setup data figures notebook app check serve
all: figures notebook app

setup:
	$(PY) -m pip install -r requirements.txt

data:            ## sanity-check the cleaned dataset
	$(PY) -c "from src.data import load_clean; d=load_clean(); print(d.shape)"

figures:         ## regenerate images/
	$(PY) scripts/make_figures.py

notebook:        ## generate and execute the analysis notebook
	$(PY) scripts/make_notebook.py
	cd notebooks && $(PY) -m jupyter nbconvert --to notebook --execute --inplace 01_plated_in_gold_analysis.ipynb

app:             ## rebuild app/data.js
	$(PY) app/build_data.py

check:           ## headless-browser smoke test + screenshots (needs requirements-dev.txt)
	$(PY) scripts/check_app.py

serve:
	cd app && $(PY) -m http.server 8000
