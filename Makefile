# make test       unit tests (no DICOM, no GPU)
# make tables     every number of the paper from data/ -> results/numbers.json, results/tables.md
# make figures    the figures of the paper -> results/fig_*.pdf
# make notebooks  write the Kaggle notebooks 04 to 08
# make smoke      end-to-end CPU run of notebooks 04 and 05 on a fake Kaggle tree in a temporary folder
.PHONY: test tables figures notebooks smoke all
test:
	python -m pytest tests/test_units.py -q
tables results/numbers.json:
	cd tools && python reproduce_tables.py
figures: results/numbers.json
	cd tools && python make_figures.py
notebooks:
	python tools/make_notebooks_v2.py
smoke:
	python tests/smoke_kaggle.py
all: test tables figures
