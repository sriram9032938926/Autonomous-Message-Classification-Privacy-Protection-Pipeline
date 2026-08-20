import os
import re
import json
import time
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

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


def load_datasets() -> pd.DataFrame:
    """Loads L1, L2, and Demo datasets in strict chronological sequence."""
    dfs = []

    # 1. Base L1 Dataset
    l1_path = "messages.csv"
    if os.path.exists(l1_path):
        df_l1 = pd.read_csv(l1_path)
        df_l1["dataset_source"] = "L1_Base"
        dfs.append(df_l1)

    # 2. L2 Development Dataset
    l2_path = os.path.join("L2_Candidate_Dataset", "l2_messages.csv")
    if os.path.exists(l2_path):
        df_l2 = pd.read_csv(l2_path)
        df_l2["dataset_source"] = "L2_Dev"
        dfs.append(df_l2)

    # 3. L2 Demo Dataset
    demo_path = os.path.join("L2_Candidate_Dataset", "l2_demo_messages.csv")
    if os.path.exists(demo_path):
        df_demo = pd.read_csv(demo_path)
        df_demo["dataset_source"] = "L2_Demo"
        dfs.append(df_demo)

    if not dfs:
        raise FileNotFoundError("No dataset files found in workspace!")

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure chronological order
    if "timestamp" in full_df.columns:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    return full_df


def run_full_l2_pipeline(custom_df: Optional[pd.DataFrame] = None) -> Tuple[
    List[Dict[str, Any]],  # Classifications
    List[Dict[str, Any]],  # Extracted Tasks/Events
    List[Dict[str, Any]],  # Sensitive Detections & Privacy Routes
    List[Dict[str, Any]],  # Related Message Groups
    List[Dict[str, Any]],  # Dynamic Priorities
    Dict[str, Any],        # Benchmark Report
    pd.DataFrame,          # Processed Full DataFrame
    IntelligentAssistant   # Query Assistant Instance
]:
    """Executes the complete end-to-end L2 pipeline deterministically."""
    
    df = custom_df if custom_df is not None else load_datasets()

    classifications = []
    extracted_items = []
    sensitive_detections = []
    privacy_routing_records = []
    processed_rows = []

    item_counter = 1

    # STEP 1: Process each message chronologically for Privacy, Classification & Extraction
    for _, row in df.iterrows():
        msg_id = str(row["message_id"]).strip()
        sender = str(row["sender"]).strip()
        text = str(row["message"]).strip()
        timestamp = str(row["timestamp"])
        source = row.get("dataset_source", "Custom")

        # 1. Privacy Detection, Masking & 3-Tier Route
        sens_info = detect_sensitive_info(msg_id, text)
        masked_text = sens_info["masked_text"] if sens_info else mask_message_text(text)
        
        route_action = sens_info["recommended_action"] if sens_info else "safe_to_process_locally"
        privacy_route = sens_info["privacy_route"] if sens_info else "safe_to_process_locally"

        if sens_info:
            sensitive_detections.append(sens_info)
            privacy_routing_records.append({
                "message_id": msg_id,
                "route": privacy_route,
                "action": route_action,
                "sensitivity_type": sens_info["sensitivity_type"],
                "risk_level": sens_info["risk"],
                "reason": sens_info["privacy_reason"],
                "masked_preview": masked_text[:60] + "..." if len(masked_text) > 60 else masked_text
            })
        else:
            privacy_routing_records.append({
                "message_id": msg_id,
                "route": "safe_to_process_locally",
                "action": "safe_to_process_locally",
                "sensitivity_type": "none",
                "risk_level": "low",
                "reason": "No sensitive credentials detected; safe for local processing.",
                "masked_preview": masked_text[:60] + "..." if len(masked_text) > 60 else masked_text
            })

        # 2. Classification
        class_res = classify_message(msg_id, sender, text, sens_info)
        classifications.append(class_res)

        # 3. Extraction of Tasks & Events
        extracted = extract_task_or_event(msg_id, class_res["category"], text, item_counter)
        if extracted:
            extracted_items.append(extracted)
            item_counter += 1

        processed_rows.append({
            "message_id": msg_id,
            "timestamp": timestamp,
            "sender": sender,
            "original_message": text,
            "masked_message": masked_text,
            "category": class_res["category"],
            "confidence": class_res["confidence"],
            "reason": class_res["reason"],
            "dataset_source": source,
            "is_sensitive": sens_info is not None,
            "sensitivity_type": sens_info["sensitivity_type"] if sens_info else None,
            "risk_level": sens_info["risk"] if sens_info else None,
            "recommended_action": route_action,
            "privacy_route": privacy_route,
            "extracted_item_id": extracted["item_id"] if extracted else None
        })

    full_df = pd.DataFrame(processed_rows)

    # STEP 2: Related-Message Grouping Engine
    grouping_engine = RelatedMessageGroupingEngine()
    groups = grouping_engine.process_message_stream(processed_rows)

    # STEP 3: Dynamic Priority & Action Engine
    priority_engine = PriorityEngine()
    priorities = priority_engine.compute_all_priorities(processed_rows, groups)

    # STEP 4: Build Intelligent Semantic Search Assistant
    assistant = IntelligentAssistant(full_df, groups, priorities, privacy_routing_records)

    # STEP 5: Run Benchmarking Comparison
    demo_queries = [
        "Which existing task became critical in the demo data?",
        "Which tasks or meetings were completed or cancelled?",
        "Which meeting was rescheduled and what is its latest schedule?",
        "Which messages contain conflicting or uncertain deadlines?",
        "Which demo messages must be blocked from external processing?",
        "Which message requires confirmation before processing?",
        "What is the latest status of the task referenced by DEMO_016?",
        "Was the compliance form approved by the finance director?"
    ]
    benchmark_report = run_system_benchmarks(full_df, assistant, demo_queries)

    # STEP 6: Save All Output JSON Artifacts
    with open("priority_output.json", "w") as f:
        json.dump(priorities, f, indent=2)

    with open("related_message_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    with open("privacy_routing_output.json", "w") as f:
        json.dump(privacy_routing_records, f, indent=2)

    with open("benchmark_comparison_report.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    # Retain L1 compatibility files
    with open("classification_results.json", "w") as f:
        json.dump(classifications, f, indent=2)

    with open("extracted_tasks_events.json", "w") as f:
        json.dump(extracted_items, f, indent=2)

    with open("sensitive_info_detections.json", "w") as f:
        json.dump(sensitive_detections, f, indent=2)

    return classifications, extracted_items, sensitive_detections, groups, priorities, benchmark_report, full_df, assistant, privacy_routing_records


# Compatibility alias for L1
def run_pipeline(csv_path: str):
    """L1 compatibility wrapper."""
    df = pd.read_csv(csv_path)
    c, e, s, g, p, b, full_df, ast, priv = run_full_l2_pipeline(df)
    return c, e, s, full_df


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("[*] EXECUTING COMPLETE L2 AUTONOMOUS PIPELINE ACROSS ALL CHRONOLOGICAL DATASETS")
    print("=" * 70)
    
    t_start = time.time()
    c, e, s, g, p, b, full_df, assistant, priv = run_full_l2_pipeline()
    t_elapsed = round(time.time() - t_start, 3)

    print(f"\n[+] Pipeline Finished in {t_elapsed} seconds!")
    print(f" • Total Processed Messages: {len(full_df)} (L1: 900, L2 Dev: 180, L2 Demo: 24)")
    print(f" • Classifications: {len(c)}")
    print(f" • Extracted Tasks & Events: {len(e)}")
    print(f" • Sensitive Detections: {len(s)}")
    print(f" • Related-Message Lifecycle Groups: {len(g)}")
    print(f" • Dynamic Priority Decisions: {len(p)}")
    print(f" • Benchmark Average Latency: {b['benchmark_summary']['optimized_avg_latency_ms']} ms ({b['benchmark_summary']['latency_reduction_factor']})")
    
    print("\n" + "=" * 70)
    print("[*] TESTING ALL 8 MANDATORY DEMO QUERIES (DQ01 - DQ08)")
    print("=" * 70)
    
    queries_file = os.path.join("L2_Candidate_Dataset", "l2_demo_queries.csv")
    if os.path.exists(queries_file):
        q_df = pd.read_csv(queries_file)
        for _, r in q_df.iterrows():
            qid = r["query_id"]
            qtxt = r["query"]
            ans = assistant.answer_query(qtxt)
            print(f"\n[{qid}] Query: {qtxt}")
            print(f" -> Answer: {ans['answer']}")
            print(f" -> Supporting IDs: {ans['supporting_message_ids']}")
            print(f" -> Reason: {ans['reason']}")
            print(f" -> Relevance Score: {ans.get('relevance_score')}")
