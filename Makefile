load_docs:
	python src/load_docs.py

serve:
	fastapi dev src/main.py
