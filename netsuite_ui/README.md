NetSuite UI (Streamlit)

Run the backend FastAPI server first (see prototype/README or run:

```bash
uvicorn prototype.server:app --reload --host 0.0.0.0 --port 8000
```

Then run the Streamlit UI:

```bash
cd netsuite_ui
streamlit run app.py
```

The UI allows uploading transcript, pm_export, template, and optional summary files, then triggering plan generation via the backend `/generate` endpoint.
