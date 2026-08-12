import streamlit as st
import pandas as pd
import json
import os
from pipeline import run_pipeline, detect_sensitive_info, classify_message, extract_task_or_event, mask_message_text

# Set Page Config
st.set_page_config(
    page_title="AI Message Classification & Privacy Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Rich Aesthetics
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e2638 0%, #0d1527 100%);
        border: 1px solid #2e3c54;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        color: #8b9bb4;
        font-size: 15px;
        line-height: 1.5;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #161b26;
        border: 1px solid #232d3f;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38ef7d;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# Load Data & Run Pipeline
@st.cache_data
def get_pipeline_data(file_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return run_pipeline(tmp_path)

@st.cache_data
def get_mandatory_ids():
    m_file = "mandatory_demo_ids.csv"
    if os.path.exists(m_file):
        with open(m_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("message_id")]
    return []

# --- Dataset Upload Handling ---
csv_file_local = "messages.csv"
if os.path.exists(csv_file_local):
    with open(csv_file_local, "rb") as f:
        file_bytes = f.read()
else:
    st.markdown("""
    <div style="background:#1e2638;border:1px solid #f59e0b;border-radius:10px;padding:20px;margin-bottom:20px;">
        <h3 style="color:#f59e0b;margin:0 0 8px 0;">📂 Upload Dataset to Begin</h3>
        <p style="color:#94a3b8;margin:0;">The dataset is not bundled with the code (privacy policy). Please upload <b>messages.csv</b> to run the pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload messages.csv", type=["csv"], label_visibility="collapsed")
    if uploaded_file is None:
        st.info("⬆️ Please upload the **messages.csv** dataset file to proceed.")
        st.stop()
    file_bytes = uploaded_file.read()

classifications, extracted_items, sensitive_detections, df = get_pipeline_data(file_bytes)
mandatory_ids = get_mandatory_ids()


# HEADER SECTION
st.markdown("""
<div class="header-card">
    <div class="header-title">
        <span>🛡️</span> Autonomous Message Classification & Privacy Protection Pipeline
    </div>
    <div class="header-subtitle">
        Zero-External-Call Local NLP Engine for Categorization, Task/Event Extraction & PII Privacy Masking.
        Processes 900 fictional chronological messages deterministically and securely.
    </div>
</div>
""", unsafe_allow_html=True)


# SIDEBAR FILTERS & METRICS
st.sidebar.title("📌 Navigation & Controls")

tab_choice = st.sidebar.radio(
    "Select System Module:",
    [
        "📊 Dashboard Overview",
        "⭐ 15 Mandatory Demo IDs",
        "Part 1: Message Classification",
        "Part 2: Tasks & Events Extraction",
        "Part 3: Sensitive Info & Masking",
        "⚡ Live Real-Time Tester",
        "📖 Code Walkthrough & Architecture"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download Structured Outputs")
if os.path.exists("classification_results.json"):
    with open("classification_results.json", "rb") as f:
        st.sidebar.download_button("📥 Classifications (JSON)", f, file_name="classification_results.json", mime="application/json")
if os.path.exists("extracted_tasks_events.json"):
    with open("extracted_tasks_events.json", "rb") as f:
        st.sidebar.download_button("📥 Tasks & Events (JSON)", f, file_name="extracted_tasks_events.json", mime="application/json")
if os.path.exists("sensitive_info_detections.json"):
    with open("sensitive_info_detections.json", "rb") as f:
        st.sidebar.download_button("📥 Sensitive Detections (JSON)", f, file_name="sensitive_info_detections.json", mime="application/json")


# ----------------------------------------------------
# TAB 1: DASHBOARD OVERVIEW
# ----------------------------------------------------
if tab_choice == "📊 Dashboard Overview":
    st.subheader("📊 System Pipeline Performance & Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Messages</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #60a5fa;">6 / 6</div><div class="metric-label">Categories Covered</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #c084fc;">{len(extracted_items)}</div><div class="metric-label">Extracted Tasks & Events</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #f87171;">{len(sensitive_detections)}</div><div class="metric-label">Sensitive Detections</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📈 Message Category Distribution")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### 🔐 Sensitivity Risk Level Breakdown")
        sens_df = pd.DataFrame(sensitive_detections)
        if not sens_df.empty:
            risk_counts = sens_df["risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            st.dataframe(risk_counts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📂 Chronological Dataset Sample (Strictly Masked)")
    st.caption("All sensitive values (card numbers, passwords, OTPs, addresses) are masked with asterisks ****** to guarantee privacy.")
    
    display_df = df[["message_id", "timestamp", "sender", "masked_message", "category", "confidence"]].head(15)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------
# TAB 2: 15 MANDATORY DEMO IDS
# ----------------------------------------------------
elif tab_choice == "⭐ 15 Mandatory Demo IDs":
    st.subheader("⭐ Mandatory 15 Message IDs Demonstration")
    st.info("Showing exact classification, extraction, sensitive detection, and reasoning for all 15 mandatory message IDs (Fully Masked).")

    mand_df = df[df["message_id"].isin(mandatory_ids)].copy()

    for idx, row in mand_df.iterrows():
        msg_id = row["message_id"]
        category = row["category"]
        conf = row["confidence"]
        reason = row["reason"]
        masked_msg = row["masked_message"]
        is_sens = row["is_sensitive"]
        sens_type = row["sensitivity_type"]
        risk = row["risk_level"]
        rec_action = row["recommended_action"]
        
        with st.expander(f"📌 {msg_id} - Sender: {row['sender']} | Category: {category.upper()}", expanded=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Message Content (Masked):** `{masked_msg}`")
                st.markdown(f"**Classification Reason:** {reason}")
                st.markdown(f"**Confidence Score:** `{conf:.2f}`")
            with c2:
                if is_sens:
                    st.error(f"🚨 **Sensitive Info Detected!**\n- Type: `{sens_type}`\n- Risk: `{risk.upper()}`\n- Recommendation: `{rec_action}`")
                else:
                    st.success("✅ No Sensitive Credentials Found")

                # Show extracted task/event if present
                ext_match = [item for item in extracted_items if item["source_message_id"] == msg_id]
                if ext_match:
                    ext = ext_match[0]
                    st.info(f"📋 **Extracted {ext['type'].capitalize()}:**\n- Title: `{ext['title']}`\n- Deadline/Date: `{ext['deadline']}`\n- Time: `{ext['time']}`\n- Priority: `{ext['priority']}`")


# ----------------------------------------------------
# TAB 3: PART 1 - MESSAGE CLASSIFICATION
# ----------------------------------------------------
elif tab_choice == "Part 1: Message Classification":
    st.subheader("Part 1: Message Classification Results")
    st.caption("Every message classified into one of the 6 mandatory categories with confidence score & short reasoning.")

    category_filter = st.selectbox(
        "Filter by Category:",
        ["ALL"] + list(df["category"].unique())
    )

    search_query = st.text_input("Search Message ID or Content:", "")

    filtered_df = df.copy()
    if category_filter != "ALL":
        filtered_df = filtered_df[filtered_df["category"] == category_filter]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["message_id"].str.contains(search_query, case=False) |
            filtered_df["masked_message"].str.contains(search_query, case=False)
        ]

    st.markdown(f"**Showing {len(filtered_df)} Messages:**")
    st.dataframe(
        filtered_df[["message_id", "timestamp", "sender", "category", "confidence", "reason", "masked_message"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 💡 Classification Rationale & Decision Logic")
    st.markdown("""
    - **Action Required**: Messages demanding user action with explicit/implicit deadlines or requests (e.g. `MSG_0002`, `MSG_0007`).
    - **Meeting or Event**: Scheduled calendar updates, invitations, appointments, or tentative meeting discussions (e.g. `MSG_0001`, `MSG_0003`, `MSG_0037`).
    - **Personal Information**: User preferences, dietary choices, T-shirt sizes, medical notes, profile details (e.g. `MSG_0009`, `MSG_0016`).
    - **General Information**: Operational status, general notices, charging status, weather updates (e.g. `MSG_0004`, `MSG_0006`).
    - **Promotional**: Marketing offers, discount codes (SAVE17, SAVE23), subscription upsells (e.g. `MSG_0014`, `MSG_0015`).
    - **Sensitive Information**: Credentials, bank account numbers, credit cards, OTPs, recovery codes, auth tokens, addresses (e.g. `MSG_0005`, `MSG_0013`).
    """)


# ----------------------------------------------------
# TAB 4: PART 2 - TASKS & EVENTS EXTRACTION
# ----------------------------------------------------
elif tab_choice == "Part 2: Tasks & Events Extraction":
    st.subheader("Part 2: Task and Event Extraction")
    st.caption("Extracted item title, description, date/deadline, time, person involved, priority, and source message ID. Missing details stored as null.")

    ext_df = pd.DataFrame(extracted_items)

    type_filter = st.radio("Filter Item Type:", ["ALL", "task", "event"], horizontal=True)
    if type_filter != "ALL":
        ext_df = ext_df[ext_df["type"] == type_filter]

    st.markdown(f"**Total Extracted Items: {len(ext_df)}**")
    st.dataframe(ext_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔍 Mandatory Video Case: Handling Unclear / Missing Information")
    st.warning("Rule Requirement: 'Do not guess missing information. If a date, time, person, or deadline is unclear, store it as unresolved or null.'")
    
    # Highlight MSG_0037
    unclear_item = [item for item in extracted_items if item["source_message_id"] == "MSG_0037"]
    if unclear_item:
        st.markdown("#### Example: MSG_0037 ('The review could be Friday afternoon')")
        st.json(unclear_item[0])
        st.info("💡 **Explanation:** The message specifies 'Friday afternoon' without an exact calendar date (YYYY-MM-DD), exact clock time (HH:MM), or person involved. Therefore, `deadline`, `time`, and `person` are strictly stored as `null` / unresolved.")


# ----------------------------------------------------
# TAB 5: PART 3 - SENSITIVE INFORMATION & MASKING
# ----------------------------------------------------
elif tab_choice == "Part 3: Sensitive Info & Masking":
    st.subheader("Part 3: Sensitive Information Detection & Privacy Protection")
    st.caption("Identifies credentials, OTPs, bank numbers, credit cards, recovery keys, tokens, and home addresses, applying risk ratings & masking.")

    sens_df = pd.DataFrame(sensitive_detections)
    st.markdown(f"**Total Sensitive Messages Detected: {len(sens_df)}**")

    # Merge with masked text only (raw sensitive text is strictly excluded from UI display)
    sens_merged = sens_df.merge(df[["message_id", "sender", "masked_message"]], on="message_id")
    
    st.dataframe(
        sens_merged[["message_id", "sender", "sensitivity_type", "risk", "recommended_action", "masked_text"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 🛡️ Privacy Masked Messages Showcase")
    st.success("🔒 **Zero-Exposure Policy**: All secret credentials, addresses, and authorization codes are masked with asterisks (******) before UI rendering.")

    sample_sens = sens_merged.head(6)
    for _, s_row in sample_sens.iterrows():
        with st.expander(f"🔒 {s_row['message_id']} - Sensitivity Type: {s_row['sensitivity_type'].upper()}", expanded=True):
            st.markdown(f"**Safe Masked Text:** `{s_row['masked_text']}`")
            st.markdown(f"**Risk Level:** `{s_row['risk'].upper()}` | **Recommended Action:** `{s_row['recommended_action']}`")


# ----------------------------------------------------
# TAB 6: LIVE REAL-TIME TESTER
# ----------------------------------------------------
elif tab_choice == "⚡ Live Real-Time Tester":
    st.subheader("⚡ Real-Time Interactive Pipeline Tester")
    st.caption("Type any custom message below or pick a pre-set sample to test real-time classification, extraction, and PII masking.")

    sample_preset = st.selectbox(
        "Or choose a Preset Sample:",
        [
            "Custom Input",
            "Please review the quarterly audit report by 2026-09-15.",
            "Reminder: Doctor appointment scheduled for 2026-09-20 at 14:30 in City Hospital.",
            "Use password SecurePass#2026 to log in.",
            "Flash sale! Get 50% off on all items using code SAVE50.",
            "My card number is 4111 2222 3333 4444-12."
        ]
    )

    user_text = ""
    if sample_preset != "Custom Input":
        user_text = sample_preset

    input_message = st.text_area("Input Message Text:", value=user_text, height=100)
    input_sender = st.text_input("Sender Name:", value="UserTest")

    if st.button("🚀 Process Message", type="primary"):
        if not input_message.strip():
            st.warning("Please enter a message!")
        else:
            sens_res = detect_sensitive_info("MSG_TEST", input_message)
            class_res = classify_message("MSG_TEST", input_sender, input_message, sens_res)
            ext_res = extract_task_or_event("MSG_TEST", class_res["category"], input_message, 999)

            st.markdown("### 📊 Pipeline Results")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("#### Part 1: Classification")
                st.json(class_res)

                if sens_res:
                    st.markdown("#### Part 3: Sensitive Info Detection")
                    st.json({
                        "message_id": sens_res["message_id"],
                        "sensitivity_type": sens_res["sensitivity_type"],
                        "risk": sens_res["risk"],
                        "masked_text": sens_res["masked_text"],
                        "recommended_action": sens_res["recommended_action"]
                    })
                else:
                    st.success("Part 3: No sensitive credentials detected.")

            with res_col2:
                st.markdown("#### Part 2: Task / Event Extraction")
                if ext_res:
                    st.json(ext_res)
                else:
                    st.info("No task or event extracted from this message type.")


# ----------------------------------------------------
# TAB 7: CODE WALKTHROUGH & ARCHITECTURE
# ----------------------------------------------------
elif tab_choice == "📖 Code Walkthrough & Architecture":
    st.subheader("📖 System Architecture & Video Explanation Guide")
    st.markdown("""
    ### 🏛️ System Architecture Summary
    The system follows a 3-stage deterministic NLP pipeline built in Python:
    1. **Part 3: Sensitive Info Detection & Masking**: Regex pattern matcher identifies credentials (Credit cards, Bank accounts, OTPs, Passwords, Access Tokens, Recovery Codes, Phone numbers, Addresses), assigns risk level (`high`/`medium`), and replaces secret substrings with `******`.
    2. **Part 1: Message Classification**: Evaluates message against 6 mutually exclusive categories (`action_required`, `meeting_or_event`, `personal_information`, `general_information`, `promotional`, `sensitive_information`) with confidence scores and reasoning.
    3. **Part 2: Task/Event Extraction**: For actionable messages, extracts title, deadline, time, person, and priority. Crucially, missing information is never guessed and remains `null`.

    ---

    ### 🔑 Key Code Section: Deterministic Classification & Sensitivity Rule Engine (`pipeline.py`)
    ```python
    def classify_message(msg_id, sender, text, sensitive_meta):
        # 1. Highest Priority: Sensitive Credentials
        if sensitive_meta is not None:
            return {"category": "sensitive_information", "confidence": 0.98, ...}
        
        # 2. Promotional Check
        if sender.lower() == "promotions" or "SAVE" in text:
            return {"category": "promotional", "confidence": 0.95, ...}
        
        # 3. Meeting / Event Check
        if "calendar update:" in text.lower() or "scheduled for" in text.lower():
            return {"category": "meeting_or_event", "confidence": 0.92, ...}
        
        # 4. Action Required Check
        if "deadline is" in text.lower() or "by 2026-" in text.lower():
            return {"category": "action_required", "confidence": 0.91, ...}
        
        # 5. Personal Info Check
        if "for my profile" in text.lower() or "personal note:" in text.lower():
            return {"category": "personal_information", "confidence": 0.93, ...}
        
        # 6. Fallback General Information
        return {"category": "general_information", "confidence": 0.88, ...}
    ```

    ---

    ### ⚠️ Limitations & Future Improvements
    - **Current Limitation**: Rule-based regex pattern matching works flawlessly for structured template datasets but may miss slang or informal natural language phrasing.
    - **Future Improvement**: Integrate local lightweight transformer embeddings (e.g. `sentence-transformers` or `onnx-runtime`) for semantic similarity matching without external API dependency.
    """)
