.PHONY: install data ingest api ui mcp test
install:        ## create venv + install deps
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:           ## (re)generate the mock corpus and MCP data
	python scripts/generate_sample_data.py
ingest:         ## build index (and push to Pinecone if PINECONE_API_KEY is set)
	python -m app.ingestion.ingest
api:            ## FastAPI on :8000
	uvicorn app.main:app --reload --port 8000
ui:             ## Streamlit on :8501
	streamlit run ui/streamlit_app.py
mcp:            ## optional: standalone MCP server (set MCP_TRANSPORT=http)
	MCP_SERVER_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 python -m app.mcp_server.server
test:
	python -m pytest -q
