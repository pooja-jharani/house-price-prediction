.PHONY: install data train tune predict test lint app clean

install:
	pip install -e ".[dev,app]"

data:
	house-price generate-data

train:
	house-price train

tune:
	house-price train --tune

predict:
	house-price predict --area 1800 --bedrooms 3 --bathrooms 2 --age 5 --distance 4.5 --location Suburb

test:
	pytest -q

lint:
	ruff check src tests app

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf models/*.joblib models/*.json reports build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
