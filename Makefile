load_docs:
	python src/load_docs.py

serve:
	fastapi dev src/api.py

ui:
	streamlit run src/main.py
