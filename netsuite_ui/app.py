import streamlit as st
import requests
import json

st.set_page_config(page_title="NetSuite UI", layout="wide")

DEFAULT_BACKEND = "http://localhost:8000"

if "file_ids" not in st.session_state:
    st.session_state.file_ids = {}


def upload_to_backend(file, endpoint_base, key_name):
    url = f"{endpoint_base}/upload/{key_name}"
    try:
        files = {"file": (file.name, file.getvalue())}
        resp = requests.post(url, files=files, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("file_id")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


st.title("NetSuite UI")

backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND)

# Health check
with st.container():
    st.subheader("Backend Health")
    try:
        r = requests.get(f"{backend_url}/health", timeout=2)
        if r.status_code == 200:
            st.success("Backend reachable")
        else:
            st.warning(f"Backend returned status {r.status_code}")
    except Exception:
        st.error("Cannot reach backend. Make sure the FastAPI server is running.")

st.markdown("---")

st.subheader("Upload Files")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    transcript = st.file_uploader("Transcript (text)", type=["txt"], key="u_transcript")
    if transcript and st.button("Upload transcript"):
        fid = upload_to_backend(transcript, backend_url, "transcript")
        if fid:
            st.session_state.file_ids["transcript"] = fid
            st.success(f"Uploaded — file_id={fid}")
    if st.session_state.file_ids.get("transcript"):
        st.write("Transcript file_id:", st.session_state.file_ids.get("transcript"))

with col2:
    pm_export = st.file_uploader("PM export (json)", type=["json"], key="u_pm")
    if pm_export and st.button("Upload PM export"):
        try:
            # quick local validation
            json.loads(pm_export.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
        else:
            fid = upload_to_backend(pm_export, backend_url, "pm_export")
            if fid:
                st.session_state.file_ids["pm_export"] = fid
                st.success(f"Uploaded — file_id={fid}")
    if st.session_state.file_ids.get("pm_export"):
        st.write("PM export file_id:", st.session_state.file_ids.get("pm_export"))

with col3:
    template = st.file_uploader("Template (json)", type=["json"], key="u_template")
    if template and st.button("Upload template"):
        try:
            json.loads(template.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
        else:
            fid = upload_to_backend(template, backend_url, "template")
            if fid:
                st.session_state.file_ids["template"] = fid
                st.success(f"Uploaded — file_id={fid}")
    if st.session_state.file_ids.get("template"):
        st.write("Template file_id:", st.session_state.file_ids.get("template"))

with col4:
    summary = st.file_uploader("Prior engagement summary (text, optional)", type=["txt"], key="u_summary")
    if summary and st.button("Upload summary"):
        fid = upload_to_backend(summary, backend_url, "summary")
        if fid:
            st.session_state.file_ids["summary"] = fid
            st.success(f"Uploaded — file_id={fid}")
    if st.session_state.file_ids.get("summary"):
        st.write("Summary file_id:", st.session_state.file_ids.get("summary"))

st.markdown("---")

st.subheader("Generate Plan")
tenant_id = st.text_input("Tenant ID", value="mytenant")
engagement_id = st.text_input("Engagement ID", value="eng123")

can_generate = all(k in st.session_state.file_ids for k in ("transcript", "pm_export", "template"))

if not can_generate:
    st.info("Please upload transcript, pm_export and template files before generating.")

if st.button("Generate Plan"):
    if not can_generate:
        st.error("Missing required uploads.")
    else:
        payload = {
            "tenant_id": tenant_id,
            "engagement_id": engagement_id,
            "transcript_file_id": st.session_state.file_ids.get("transcript"),
            "pm_export_file_id": st.session_state.file_ids.get("pm_export"),
            "template_file_id": st.session_state.file_ids.get("template"),
        }
        if st.session_state.file_ids.get("summary"):
            payload["summary_file_id"] = st.session_state.file_ids.get("summary")

        with st.spinner("Generating plan — this may take a while..."):
            try:
                resp = requests.post(f"{backend_url}/generate", data=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                plan = data.get("plan") or data
                st.success("Plan generated")
                #st.json(plan)
                st.markdown(f"```json\n{json.dumps(plan, indent=2)}\n```")
                if isinstance(plan, dict):
                    for key, value in plan.items():
                        st.markdown(f"### {key.capitalize()}")
                        st.markdown(f"{value}")

                # download
                st.download_button(
                    label="Download plan JSON",
                    data=json.dumps(plan, indent=2),
                    file_name=f"draft_plan_{engagement_id}.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

st.markdown("---")

if st.button("Clear uploaded file IDs"):
    st.session_state.file_ids = {}
    st.success("Cleared.")
