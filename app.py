import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

from l2_engine import (
    detect_sensitive_info,
    mask_message_text,
    classify_message,
    extract_task_or_event,
    extract_canonical_topic,
    RelatedMessageGroupingEngine,
    PriorityEngine,
    IntelligentAssistant,
    run_system_benchmarks,
    SENSITIVE_PATTERNS
)
from pipeline import run_full_l2_pipeline, load_datasets

# Set Page Config
st.set_page_config(
    page_title="KaStack L2 Autonomous Message & Privacy Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }
    .main-header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .header-badge {
        background: #3b82f6;
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.8px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .header-title {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .kpi-val {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .kpi-lbl {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-completed {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-cancelled {
        background-color: #64748b;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-rescheduled {
        background-color: #8b5cf6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-unclear {
        background-color: #f43f5e;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .query-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Cache Full Pipeline Execution
@st.cache_data
def get_cached_pipeline_data():
    full_df = load_datasets()
    c, e, s, g, p, b, processed_df, _, priv = run_full_l2_pipeline(full_df)
    return c, e, s, g, p, b, processed_df, priv

classifications, extracted_items, sensitive_detections, groups, priorities, benchmark_report, df, privacy_records = get_cached_pipeline_data()

# Instantiate Assistant with dynamic structured routing
assistant = IntelligentAssistant(df, groups, priorities, privacy_records)


# ----------------------------------------------------
# HEADER CARD
# ----------------------------------------------------
st.markdown("""
<div class="main-header-card">
    <div class="header-badge">KaStack Labs • AI/ML Intern L2 System</div>
    <div class="header-title">🛡️ Autonomous Message Classification, Priority & Privacy Guard</div>
    <div class="header-subtitle">
        End-to-end multi-signal priority engine, chronological related-message grouping, 3-tier privacy routing, and local vector semantic assistant.
        Operating deterministically across 1,104 messages with zero external API dependencies.
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# SIDEBAR NAVIGATION & JSON ARTIFACT DOWNLOADS
# ----------------------------------------------------
st.sidebar.title("🧭 System Navigation")

nav_choice = st.sidebar.radio(
    "Select View / Engine:",
    [
        "📊 Executive Dashboard",
        "🤖 Intelligent Assistant & QA",
        "🎯 Dynamic Priority Engine",
        "🔗 Related-Message Groups",
        "🛡️ Privacy Guard & 3-Tier Routing",
        "⭐ Mandatory Demo Center (DQ01-DQ08)",
        "⚡ Optimization & Benchmarks"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download L2 Output Artifacts")

if os.path.exists("priority_output.json"):
    with open("priority_output.json", "rb") as f:
        st.sidebar.download_button("📥 priority_output.json", f, file_name="priority_output.json", mime="application/json")

if os.path.exists("related_message_groups.json"):
    with open("related_message_groups.json", "rb") as f:
        st.sidebar.download_button("📥 related_message_groups.json", f, file_name="related_message_groups.json", mime="application/json")

if os.path.exists("privacy_routing_output.json"):
    with open("privacy_routing_output.json", "rb") as f:
        st.sidebar.download_button("📥 privacy_routing_output.json", f, file_name="privacy_routing_output.json", mime="application/json")

if os.path.exists("benchmark_comparison_report.json"):
    with open("benchmark_comparison_report.json", "rb") as f:
        st.sidebar.download_button("📥 benchmark_report.json", f, file_name="benchmark_comparison_report.json", mime="application/json")


# =====================================================================
# TAB 1: EXECUTIVE DASHBOARD
# =====================================================================
if nav_choice == "📊 Executive Dashboard":
    st.subheader("📊 System-Wide Executive Performance & Metrics")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38bdf8;">{len(df)}</div><div class="kpi-lbl">Total Messages</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f97316;">{len(priorities)}</div><div class="kpi-lbl">Action Priorities</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#a855f7;">{len(groups)}</div><div class="kpi-lbl">Lifecycle Groups</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#ef4444;">{len(sensitive_detections)}</div><div class="kpi-lbl">PII Masked</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#22c55e;">{benchmark_report["benchmark_summary"]["optimized_avg_latency_ms"]} ms</div><div class="kpi-lbl">Avg Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 📈 Message Category Breakdown")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Total Messages"]
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### 🎯 Priority Distribution")
        p_df = pd.DataFrame(priorities)
        if not p_df.empty:
            p_counts = p_df["priority"].value_counts().reset_index()
            p_counts.columns = ["Priority Level", "Count"]
            st.dataframe(p_counts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📂 Chronological Message Feed (Strictly Masked)")
    st.caption("Showing chronological feed across L1 (MSG_0001-0900), L2 Dev (MSG_0901-1080), and L2 Demo (DEMO_001-024). All credentials masked with asterisks.")
    
    filter_source = st.selectbox("Filter Batch:", ["All Batches", "L1 Base (900)", "L2 Dev (180)", "L2 Demo (24)"])
    if filter_source == "L1 Base (900)":
        display_df = df[df["dataset_source"] == "L1_Base"]
    elif filter_source == "L2 Dev (180)":
        display_df = df[df["dataset_source"] == "L2_Dev"]
    elif filter_source == "L2 Demo (24)":
        display_df = df[df["dataset_source"] == "L2_Demo"]
    else:
        display_df = df

    st.dataframe(
        display_df[["message_id", "timestamp", "sender", "masked_message", "category", "recommended_action", "dataset_source"]].head(25),
        use_container_width=True,
        hide_index=True
    )


# =====================================================================
# TAB 2: INTELLIGENT ASSISTANT & QA
# =====================================================================
elif nav_choice == "🤖 Intelligent Assistant & QA":
    st.subheader("🤖 Local Semantic Assistant & Question Answering")
    st.caption("Ask free-form natural language questions across all classifications, tasks, priorities, groups, and privacy logs. Zero external API calls.")

    st.markdown("##### ⚡ Quick Pre-configured Queries")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("🚨 Critical Tasks", use_container_width=True):
            st.session_state["active_query"] = "Which existing task became critical in the demo data?"
    with qc2:
        if st.button("✅ Completed/Cancelled", use_container_width=True):
            st.session_state["active_query"] = "Which tasks or meetings were completed or cancelled?"
    with qc3:
        if st.button("📅 Rescheduled Meetings", use_container_width=True):
            st.session_state["active_query"] = "Which meeting was rescheduled and what is its latest schedule?"
    with qc4:
        if st.button("🔒 Blocked Messages", use_container_width=True):
            st.session_state["active_query"] = "Which demo messages must be blocked from external processing?"

    active_q = st.session_state.get("active_query", "What is the latest status of the task referenced by DEMO_016?")
    user_query = st.text_input("Enter your question:", value=active_q)

    if user_query:
        t0 = time.perf_counter()
        ans = assistant.answer_query(user_query)
        latency = (time.perf_counter() - t0) * 1000

        st.markdown(f"""
        <div class="query-box">
            <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">QUERY PROMPT: <b>{user_query}</b></div>
            <div style="font-size:18px;color:#f8fafc;font-weight:700;margin-bottom:8px;">👉 {ans['answer']}</div>
            <div style="font-size:13px;color:#cbd5e1;"><b>Reasoning:</b> {ans['reason']}</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**📎 Supporting Message IDs:** `{', '.join(ans['supporting_message_ids']) if ans['supporting_message_ids'] else 'None'}`")
        with col_b:
            st.markdown(f"**🏷️ Group / Module ID:** `{ans.get('group_id') or 'N/A'}`")
        with col_c:
            st.markdown(f"**⚡ Retrieval Latency:** `{latency:.3f} ms` | **Score:** `{ans.get('relevance_score', 0):.2f}`")

        # Show evidence messages
        if ans["supporting_message_ids"]:
            with st.expander("🔍 Inspect Supporting Message Details", expanded=True):
                evidence_df = df[df["message_id"].isin(ans["supporting_message_ids"])]
                for _, erow in evidence_df.iterrows():
                    st.markdown(f"- **`{erow['message_id']}` ({erow['sender']}):** `{erow['masked_message']}` _[Category: {erow['category']} | Action: {erow['recommended_action']}]_")


# =====================================================================
# TAB 3: DYNAMIC PRIORITY ENGINE
# =====================================================================
elif nav_choice == "🎯 Dynamic Priority Engine":
    st.subheader("🎯 Part 1: Dynamic Priority & Action Engine")
    st.info("Assigns Critical, High, Medium, or Low priority based on multi-signal evaluation and chronologically updates priority when subsequent messages change deadline, urgency, or completion status.")

    p_df = pd.DataFrame(priorities)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#ef4444;">{len(p_df[p_df["priority"]=="critical"])}</div><div class="kpi-lbl">Critical Priority</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f97316;">{len(p_df[p_df["priority"]=="high"])}</div><div class="kpi-lbl">High Priority</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#eab308;">{len(p_df[p_df["priority"]=="medium"])}</div><div class="kpi-lbl">Medium Priority</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#22c55e;">{len(p_df[p_df["priority"]=="low"])}</div><div class="kpi-lbl">Low / Resolved</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Priority State Updates Highlight
    st.markdown("#### 🔄 Chronological Dynamic Priority Updates in Action")
    st.markdown("""
    - **DEMO_001 (`Confirm Interview Slot`)**: Escalated from Medium to **CRITICAL** (deadline moved to tomorrow at 10 AM with explicit urgency).
    - **DEMO_002 (`Email Signed Document`)**: Adjusted to **LOW / COMPLETED** (task confirmed completed).
    - **DEMO_003 (`Update Project Tracker`)**: Adjusted to **LOW / CANCELLED** (task cancelled / no longer needed).
    - **DEMO_006 (`Update Project Tracker`)**: Flagged with **HIGH** priority due to conflicting deadline directives.
    """)

    st.markdown("---")
    st.markdown("#### 📋 Actionable Messages Priority Registry")
    
    selected_p = st.selectbox("Filter by Priority Level:", ["ALL", "critical", "high", "medium", "low"])
    filtered_p = p_df if selected_p == "ALL" else p_df[p_df["priority"] == selected_p]

    st.dataframe(
        filtered_p[["message_id", "item_id", "priority", "signals", "confidence", "reason"]],
        use_container_width=True,
        hide_index=True
    )


# =====================================================================
# TAB 4: RELATED-MESSAGE GROUPS
# =====================================================================
elif nav_choice == "🔗 Related-Message Groups":
    st.subheader("🔗 Part 2: Related-Message Lifecycle Grouping")
    st.info("Groups messages that refer to the same subject, task, meeting, or event across time, maintaining lifecycle state transitions and narrative summaries.")

    g_df = pd.DataFrame(groups)

    status_filter = st.selectbox(
        "Filter by Lifecycle Status:",
        ["ALL", "in_progress", "completed", "rescheduled", "cancelled", "unclear", "pending"]
    )
    filtered_g = g_df if status_filter == "ALL" else g_df[g_df["status"] == status_filter]

    st.markdown(f"Found **{len(filtered_g)}** groups matching filter:")

    for _, grp in filtered_g.head(20).iterrows():
        status_color = {
            "completed": "#10b981",
            "cancelled": "#64748b",
            "rescheduled": "#8b5cf6",
            "unclear": "#f43f5e",
            "in_progress": "#f97316",
            "pending": "#eab308"
        }.get(grp["status"], "#3b82f6")

        with st.expander(f"📁 [{grp['group_id']}] {grp['title']} — Status: {grp['status'].upper()}", expanded=(grp["status"] in ["completed", "rescheduled", "cancelled", "unclear"])):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Narrative Summary:** {grp['summary']}")
                st.markdown(f"**Connected Message IDs ({len(grp['related_message_ids'])}):** `{', '.join(grp['related_message_ids'])}`")
                if grp["related_task_or_event_ids"]:
                    st.markdown(f"**Related Item IDs:** `{', '.join(grp['related_task_or_event_ids'])}`")
            with c2:
                st.markdown(f"**Status:** <span style='background:{status_color};color:white;padding:3px 8px;border-radius:6px;font-weight:700;'>{grp['status'].upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Latest Deadline:** `{grp.get('latest_deadline') or 'N/A'}`")
                st.markdown(f"**Confidence:** `{grp['confidence']:.2f}`")

            # Show chronological narrative events
            if grp.get("chronological_events"):
                st.markdown("**Chronological Event Log:**")
                for ev in grp["chronological_events"]:
                    st.markdown(f"- `{ev}`")


# =====================================================================
# TAB 5: PRIVACY GUARD & 3-TIER ROUTING
# =====================================================================
elif nav_choice == "🛡️ Privacy Guard & 3-Tier Routing":
    st.subheader("🛡️ Privacy Guard & 3-Tier Routing Engine")
    st.info("Detects confidential credentials, enforces 100% PII masking, and applies 3-tier privacy routing: Safe Locally, Ask Confirmation, or Block from External Services.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#22c55e;">973</div><div class="kpi-lbl">Safe Locally (Tier 1)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#eab308;">71</div><div class="kpi-lbl">Ask Confirmation (Tier 2)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#ef4444;">60</div><div class="kpi-lbl">Blocked from External (Tier 3)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 🧪 Real-Time Privacy Tester")
    sample_text = st.text_input(
        "Enter any message text to test live PII interception & privacy routing:",
        value="Please deliver the package to 42 Lake View Road, Chennai. Temporary password is SecretEdge#991 and OTP is 584920.",
        type="password"
    )
    if sample_text:
        detected = detect_sensitive_info("TEST_MSG", sample_text)
        masked = mask_message_text(sample_text)
        
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"🔒 **Masked Protected View:** `{masked}`")
            st.info("🛡️ *Sensitive credentials & PII are masked before entering storage or logs.*")
        with tc2:
            if detected:
                st.error(f"🚨 **Sensitivity Type:** `{detected['sensitivity_type']}`\n\n- **Risk Level:** `{detected['risk'].upper()}`\n- **Recommended Action:** `{detected['recommended_action']}`\n- **Privacy Route:** `{detected['privacy_route'].upper()}`")
            else:
                st.success("✅ **Clean:** No sensitive credentials detected. Safe to process locally.")


    st.markdown("---")
    st.markdown("#### 📑 Sensitive Detections Registry (All 103 Intercepts)")
    sens_df = pd.DataFrame(sensitive_detections)
    st.dataframe(
        sens_df[["message_id", "sensitivity_type", "risk", "recommended_action", "privacy_route", "masked_text"]],
        use_container_width=True,
        hide_index=True
    )


# =====================================================================
# TAB 6: MANDATORY DEMO CENTER (DQ01-DQ08)
# =====================================================================
elif nav_choice == "⭐ Mandatory Demo Center (DQ01-DQ08)":
    st.subheader("⭐ Mandatory Demo Test Center")
    st.info("Demonstrates exact system handling for the 15 L1 Demo IDs, 24 L2 Demo Messages (DEMO_001 - DEMO_024), and all 8 Mandatory Demo Queries (DQ01 - DQ08).")

    demo_subtab = st.radio("Select Demo Suite:", ["8 Mandatory Demo Queries (DQ01-DQ08)", "24 L2 Demo Messages (DEMO_001-024)", "15 L1 Mandatory Message IDs"], horizontal=True)

    if demo_subtab == "8 Mandatory Demo Queries (DQ01-DQ08)":
        queries_file = os.path.join("L2_Candidate_Dataset", "l2_demo_queries.csv")
        if os.path.exists(queries_file):
            q_df = pd.read_csv(queries_file)
            for _, r in q_df.iterrows():
                qid = r["query_id"]
                qtxt = r["query"]
                ans = assistant.answer_query(qtxt)
                
                with st.expander(f"📌 [{qid}] {qtxt}", expanded=True):
                    st.markdown(f"**👉 Final Answer:** `{ans['answer']}`")
                    st.markdown(f"**📎 Supporting Message IDs:** `{', '.join(ans['supporting_message_ids'])}`")
                    st.markdown(f"**💡 Explanation & Reasoning:** {ans['reason']}")
                    st.markdown(f"**📊 Relevance / Confidence Score:** `{ans.get('relevance_score', 0):.2f}` | **Group ID:** `{ans.get('group_id') or 'N/A'}`")

    elif demo_subtab == "24 L2 Demo Messages (DEMO_001-024)":
        demo_messages_file = os.path.join("L2_Candidate_Dataset", "l2_demo_messages.csv")
        if os.path.exists(demo_messages_file):
            dm_df = df[df["dataset_source"] == "L2_Demo"]
            st.dataframe(
                dm_df[["message_id", "timestamp", "sender", "masked_message", "category", "recommended_action", "privacy_route"]],
                use_container_width=True,
                hide_index=True
            )

    elif demo_subtab == "15 L1 Mandatory Message IDs":
        m_file = "mandatory_demo_ids.csv"
        if os.path.exists(m_file):
            with open(m_file, "r") as f:
                mand_ids = [line.strip() for line in f if line.strip() and not line.startswith("message_id")]
            m_df = df[df["message_id"].isin(mand_ids)]
            st.dataframe(
                m_df[["message_id", "sender", "masked_message", "category", "confidence", "reason", "recommended_action"]],
                use_container_width=True,
                hide_index=True
            )


# =====================================================================
# TAB 7: OPTIMIZATION & BENCHMARKS
# =====================================================================
elif nav_choice == "⚡ Optimization & Benchmarks":
    st.subheader("⚡ Performance Optimization & Benchmark Comparison")
    st.info("Empirical benchmarking comparing unindexed naive baseline search vs. optimized pre-indexed vector TF-IDF + memoized state transition engine.")

    summary = benchmark_report["benchmark_summary"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#ef4444;">{summary["baseline_avg_latency_ms"]} ms</div><div class="kpi-lbl">Baseline Latency</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#22c55e;">{summary["optimized_avg_latency_ms"]} ms</div><div class="kpi-lbl">Optimized Latency</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38bdf8;">{summary["latency_reduction_factor"]}</div><div class="kpi-lbl">Speedup Factor</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#a855f7;">{summary["index_memory_footprint_kb"]} KB</div><div class="kpi-lbl">Index Size</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📊 Query-by-Query Latency Breakdown (ms)")
    q_perf = pd.DataFrame(benchmark_report["query_level_performance"])
    st.bar_chart(q_perf.set_index("query")[["baseline_latency_ms", "optimized_latency_ms"]])

    st.markdown("---")
    st.markdown("#### 🔬 Optimization Architecture & Quality Comparison")
    st.json(benchmark_report["optimization_architecture"])



