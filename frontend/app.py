"""
Frontend Product Layer (PRD Section 5.2 - "Product Surfaces" and 7.6 - "Frontend Product Approach")

Redesigned styling: moved away from Streamlit's default red accent to a
teal/indigo palette (calmer, more "professional SaaS" feeling), improved
spacing, softer card shadows, and consistent typography scale.

Run with:  streamlit run frontend/app.py
(Make sure the backend is running first: uvicorn backend.api.main:app --reload)
"""

import streamlit as st
import requests

API_BASE = "https://coursera-multimodal-mini.onrender.com"

st.set_page_config(
    page_title="Coursera Multimodal Intelligence Platform",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #131a2b 0%, #0b0f19 55%);
    }

    .hero-banner {
        background: linear-gradient(120deg, #0ea5a3 0%, #4338ca 100%);
        padding: 2.4rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(67, 56, 202, 0.25);
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.15rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.82);
        margin: 0.5rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 400;
    }

    h3 { letter-spacing: -0.01em; }

    .insight-card {
        background: #131826;
        border: 1px solid #232a3d;
        border-left: 4px solid #14b8a6;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .insight-card.medium { border-left-color: #eab308; }
    .insight-card.low { border-left-color: #f43f5e; }

    .recommendation-box {
        background: #101a2e;
        border: 1px solid #2b3a5c;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .recommendation-box strong { color: #7dd3fc; }

    div[data-testid="stExpander"] {
        background-color: #131826;
        border: 1px solid #232a3d;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: #131826;
        border: 1px solid #232a3d;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    div[data-testid="stMetricValue"] {
        color: #14b8a6;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        border: none;
        transition: transform 0.1s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); }

    button[kind="primary"] {
        background: linear-gradient(120deg, #14b8a6 0%, #4338ca 100%) !important;
        color: white !important;
    }

    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #14b8a6 !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 22px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #131826 !important;
        color: #14b8a6 !important;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #2b3348 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1>🎓 Coursera Multimodal Intelligence Platform</h1>
    <p>Cross-modal RAG for learner friction detection — video, slides, quizzes &amp; discussions, unified.</p>
</div>
""", unsafe_allow_html=True)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

tab_upload, tab1, tab2, tab3 = st.tabs(
    ["📤  Upload Data", "🔍  Query Workspace", "✅  Review Workspace", "📊  Dashboard"]
)

# ---------------------------------------------------------------
# TAB 0: UPLOAD DATA (NEW)
# ---------------------------------------------------------------
with tab_upload:
    st.subheader("Upload learning content")
    st.caption("Supported files: .txt, .pdf, .csv, .json")

    AUTO_LABEL = "🤖 Auto-detect from file type"
    modality_options = {
        AUTO_LABEL: None,
        "🎥 Video transcript": "video",
        "📊 Slides": "slide",
        "📝 Quiz": "quiz",
        "💬 Discussion": "discussion",
    }
    # Used when Auto-detect is selected
    auto_by_ext = {"txt": "video", "pdf": "slide", "csv": "quiz", "json": "discussion"}
    st.caption("Auto-detect: .txt → Video transcript, .pdf → Slides, .csv → Quiz, .json → Discussion")

    col_a, col_b = st.columns(2)
    with col_a:
        modality_label = st.selectbox("Content type", list(modality_options.keys()))
    with col_b:
        source_title = st.text_input("Source title (optional)", placeholder="e.g. ML Week 3 Lecture")

    files = st.file_uploader(
        "Upload files",
        type=["txt", "pdf", "csv", "json"],
        accept_multiple_files=True,
    )

    if st.button("📥 Ingest files", type="primary"):
        if not files:
            st.warning("Please choose at least one file.")
        else:
            with st.spinner("Chunking and embedding... (calls Gemini, may take a while)"):
                for f in files:
                    chosen_modality = modality_options[modality_label]
                    if chosen_modality is None:
                        ext = f.name.rsplit(".", 1)[-1].lower()
                        chosen_modality = auto_by_ext.get(ext, "video")
                    try:
                        r = requests.post(
                            f"{API_BASE}/api/ingest",
                            files={"file": (f.name, f.getvalue())},
                            data={
                                "modality": chosen_modality,
                                "source_title": source_title or f.name,
                            },
                            timeout=180,
                        )
                        r.raise_for_status()
                        d = r.json()
                        st.success(
                            f"{f.name} [{chosen_modality}]: {d['chunks_indexed']} chunks indexed "
                            f"(total in DB: {d['collection_total']})"
                        )
                    except requests.exceptions.HTTPError:
                        try:
                            detail = r.json().get("detail", r.text)
                        except Exception:
                            detail = r.text
                        st.error(f"{f.name}: {detail}")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")

# ---------------------------------------------------------------
# TAB 1: QUERY WORKSPACE
# ---------------------------------------------------------------
with tab1:
    st.subheader("Ask a question about learner friction")

    col_input, col_slider = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Educator question",
            placeholder="e.g. Why are students confused about learning rate?",
            label_visibility="collapsed",
        )
    with col_slider:
        top_k = st.slider("Evidence pieces", min_value=1, max_value=10, value=5, label_visibility="collapsed")
        st.caption(f"Retrieving top {top_k} evidence pieces")

    if st.button("🚀 Run Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Retrieving evidence and synthesizing insight... (calls Gemini)"):
                try:
                    response = requests.post(
                        f"{API_BASE}/api/query",
                        json={"query": query, "top_k": top_k},
                        timeout=60,
                    )
                    response.raise_for_status()
                    st.session_state.last_result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not reach the backend. Is it running? "
                        "Start it with: uvicorn backend.api.main:app --reload"
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    result = st.session_state.last_result
    if result:
        insight = result["insight"]
        evidence = result["evidence"]

        st.divider()

        if insight.get("error") == "insufficient_evidence":
            st.warning(insight.get("confidence_reason", "No relevant evidence found."))
        elif insight.get("error"):
            st.error(f"Could not generate a grounded insight: {insight.get('confidence_reason', insight['error'])}")
        else:
            confidence = insight.get("confidence", "unknown")
            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            card_class = confidence if confidence in ("medium", "low") else ""

            st.markdown(f"""
            <div class="insight-card {card_class}">
                <h3 style="margin-top:0;">{confidence_color} Insight <span style="font-size:0.72em; color:#94a3b8; font-weight:400;">(confidence: {confidence})</span></h3>
                <p style="font-size:1.05rem; line-height:1.65; color:#e2e8f0;">{insight.get("insight", "No insight generated.")}</p>
                <p style="color:#94a3b8; font-size:0.9rem;"><em>{insight.get("confidence_reason", "")}</em></p>
                <div class="recommendation-box">
                    <strong>💡 Recommendation:</strong><br>
                    <span style="color:#e2e8f0;">{insight.get("recommendation", "No recommendation available.")}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"Query log ID: `{result['query_log_id']}` (used in Review Workspace tab)")

            st.markdown("### 📎 Evidence Panel")
            for e in evidence:
                modality_icon = {
                    "video": "🎥", "slide": "📊", "quiz": "📝", "discussion": "💬"
                }.get(e["modality"], "📄")

                used_flag = "✅ used in insight" if e["segment_id"] in insight.get("evidence_used", []) else ""

                with st.expander(
                    f"{modality_icon} {e['modality'].upper()} — {e['source_title']} "
                    f"{'@ ' + e['timestamp'] if e.get('timestamp') else ''}  {used_flag}"
                ):
                    st.write(e["text"])
                    st.caption(f"Segment ID: {e['segment_id']} | Relevance distance: {e['distance']:.3f}")

# ---------------------------------------------------------------
# TAB 2: REVIEW WORKSPACE
# ---------------------------------------------------------------
with tab2:
    st.subheader("Review the most recent insight")

    result = st.session_state.last_result
    if not result or result["insight"].get("error"):
        st.info("Run a query in the Query Workspace tab first, then come back here to review it.")
    else:
        st.markdown(f"""
        <div class="insight-card">
            <p style="color:#e2e8f0;"><strong>Query log ID:</strong> <code>{result['query_log_id']}</code></p>
            <p style="color:#e2e8f0;"><strong>Insight:</strong> {result['insight'].get('insight')}</p>
        </div>
        """, unsafe_allow_html=True)

        reviewer_note = st.text_area("Reviewer note (optional)", placeholder="Any comments for the content team...")

        col1, col2, col3 = st.columns(3)
        decision = None
        if col1.button("✅ Approve", use_container_width=True):
            decision = "approved"
        if col2.button("🔁 Needs Revision", use_container_width=True):
            decision = "needs_revision"
        if col3.button("❌ Reject", use_container_width=True):
            decision = "rejected"

        if decision:
            try:
                response = requests.post(
                    f"{API_BASE}/api/review-feedback",
                    json={
                        "query_log_id": result["query_log_id"],
                        "decision": decision,
                        "reviewer_note": reviewer_note,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                st.success(f"Decision recorded: {decision}")
            except Exception as e:
                st.error(f"Could not record decision: {e}")

# ---------------------------------------------------------------
# TAB 3: DASHBOARD
# ---------------------------------------------------------------
with tab3:
    st.subheader("Usage Metrics")

    if st.button("🔄 Refresh metrics"):
        pass

    try:
        response = requests.get(f"{API_BASE}/api/metrics", timeout=10)
        response.raise_for_status()
        metrics = response.json()

        col1, col2 = st.columns(2)
        col1.metric("Total Queries", metrics["total_queries"])
        col2.metric("Total Reviews", metrics["total_reviews"])

        st.markdown("**Confidence breakdown**")
        st.bar_chart(metrics["confidence_breakdown"], color="#14b8a6")

        st.markdown("**Reviewer decision breakdown**")
        st.bar_chart(metrics["decision_breakdown"], color="#4338ca")

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is it running?")
    except Exception as e:
        st.error(f"Something went wrong: {e}")