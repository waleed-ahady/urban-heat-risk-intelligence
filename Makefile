.PHONY: install demo train refresh app test lint validate-profile clean

install:
	python -m pip install -e ".[dev]"

demo:
	PYTHONPATH=src python scripts/bootstrap_demo.py

train:
	PYTHONPATH=src python scripts/train_models.py

refresh:
	PYTHONPATH=src python scripts/refresh_forecast.py --mode live

app:
	streamlit run app/Home.py

test:
	PYTHONPATH=src pytest

lint:
	ruff check .

validate-profile:
	PYTHONPATH=src python scripts/validate_district_profiles.py

clean:
	rm -f artifacts/*.joblib artifacts/*.json data/processed/*.csv
