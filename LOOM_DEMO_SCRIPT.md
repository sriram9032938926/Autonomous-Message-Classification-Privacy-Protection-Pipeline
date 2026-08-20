# 🎬 5-Minute Loom Video Demonstration Script & Guide

> **Target Duration:** Exactly 4:30 – 5:00 Minutes  
> **Platform:** Loom (Desktop / Screen Recording with Camera Bubble)  
> **Goal:** Showcase L2 features, explain architecture decisions, demonstrate mandatory queries, and discuss optimization/learnings.

---

## ⏱️ Minute-by-Minute Speaking Timeline

```
+-----------------------------------------------------------------------------------------------+
|  0:00 - 1:00  |  Introduction, L1 Overview & How L2 Extends L1                                |
|  1:00 - 2:00  |  Part 1 & 2: Dynamic Priority Engine & Related-Message Grouping Threads       |
|  2:00 - 3:00  |  Part 4: 3-Tier Privacy Routing & Strict PII Masking Demonstration            |
|  3:00 - 4:00  |  Part 3: Intelligent Assistant Demo & Mandatory Queries (DQ01, DQ03, DQ08)    |
|  4:00 - 5:00  |  Part 5: Optimization Benchmark (5000x Speedup), Edge Case & Wrap-up          |
+-----------------------------------------------------------------------------------------------+
```

---

### 🎙️ Minute 0:00 – 1:00 | Introduction & L1 to L2 Evolution
- **Visual:** Open Streamlit Dashboard (`Executive Dashboard` tab) showing the header card and KPI cards (1,104 total messages, 493 priority items, 95 semantic group threads, 103 masked PII).
- **What to Say:**
  > *"Hi everyone, welcome to the demonstration of my L2 submission for the AI/ML Engineer Intern role at KaStack Labs. In L1, we built an offline system for message classification, basic task extraction, and regex PII masking across 900 messages.  
  > In L2, we extended this into a stateful, autonomous assistant that processes 1,104 chronological messages across L1, L2 Development, and L2 Demo datasets.  
  > Specifically, L2 introduces: dynamic priority propagation, sentence-transformer dense semantic grouping, an offline intelligent assistant, 3-tier privacy routing, and a 5,000x optimized retrieval engine—running 100% locally with zero external API calls."*

---

### 🎙️ Minute 1:00 – 2:00 | Dynamic Priority Engine & Message Grouping
- **Visual:** Navigate to `🎯 Dynamic Priority Engine` and `🔗 Related-Message Groups` tabs.
- **What to Say:**
  > *"Let’s look at Part 1 and Part 2. Rather than using static heuristics or single keywords, our Priority Engine evaluates deadline proximity, urgency follow-ups, sender authority, and status updates dynamically across time.*  
  > *For Part 2, we use a local **Sentence-Transformer model (`all-MiniLM-L6-v2`)** to encode messages into 384-dimensional dense vectors and cluster them using **cosine similarity against running group centroids** (threshold 0.55).*  
  > *For example, in the **Related-Message Groups** tab, look at thread `GROUP_027` for 'Confirm Interview Slot':*
  > 1. *It clustered messages across time including `MSG_0027`, `DEMO_001`, and `DEMO_016` with a high semantic confidence of 0.72.*
  > 2. *When `DEMO_001` arrived stating 'The deadline to confirm the interview slot is now tomorrow at 10 AM. This is urgent', the priority engine immediately escalated the task state to **CRITICAL**.*
  > 3. *Later, `DEMO_016` noted that completion is unconfirmed, so the group status updated to **UNCLEAR**, keeping the high priority active.*
  > *Similarly, in thread `GROUP_066` ('Email Signed Document'), when `DEMO_002` confirmed completion, the status transitioned to **COMPLETED** and active urgency was automatically cleared."*

---

### 🎙️ Minute 2:00 – 3:00 | 3-Tier Privacy Guard & Masking Demonstration
- **Visual:** Navigate to `🛡️ Privacy Guard & 3-Tier Routing` tab. Show the 3 KPI cards (973 Safe Locally, 71 Ask Confirmation, 60 Blocked) and the **Real-Time Privacy Tester**.
- **What to Say:**
  > *"Data privacy and protection are fundamental to our architecture. We implemented a 3-Tier Privacy Routing Guard:*
  > 1. ***Safe Locally (Tier 1)***: *Standard operational text and masked phone/address notes (e.g. `DEMO_014` / `MSG_0005`).*
  > 2. ***Ask Confirmation (Tier 2)***: *Government IDs and confidential medical notes. For instance, `DEMO_015` mentions a 'vitamin B12 deficiency' and is flagged for explicit user confirmation before processing.*
  > 3. ***Blocked from External Services (Tier 3)***: *High-risk security credentials like One-Time Passwords in `DEMO_012`, passwords in `DEMO_013`, and API tokens in `DEMO_024` are strictly blocked from external transmission.*
  > *Most importantly, all sensitive values are 100% masked with asterisks across every UI view, output file, and retrieval index."*

---

### 🎙️ Minute 3:00 – 4:00 | Semantic Assistant & Mandatory Demo Queries
- **Visual:** Navigate to `🤖 Intelligent Assistant & QA` (or `⭐ Mandatory Demo Center`).
- **What to Say & Click:**
  > *"Now let's test our local Semantic Assistant. It combines structured upstream resolution with offline dense sentence embeddings to provide grounded answers with exact supporting message IDs and reasons.*
  > *Let's run the mandatory demo queries:*
  > - ***Query DQ01***: *'Which existing task became critical in the demo data?' -> Correctly identifies 'Confirm the interview slot' with supporting IDs `DEMO_001`, `MSG_0027`, etc.*
  > - ***Query DQ03***: *'Which meeting was rescheduled and what is its latest schedule?' -> Correctly tracks 'Internship Orientation' through `DEMO_007`, `DEMO_009`, and `DEMO_017` to its latest confirmed time: **2026-10-07 at 17:30**.*
  > - ***Query DQ08***: *'Was the compliance form approved by the finance director?' -> Look at how the assistant handles this: it explicitly retrieves the closest candidate message (`DEMO_022`) but makes zero unsupported claims, maintaining 100% grounding without hallucination."*


---

### 🎙️ Minute 4:00 – 5:00 | Benchmarking, Edge Cases & Reflection
- **Visual:** Navigate to `⚡ Optimization & Benchmarks` tab showing the comparison chart.
- **What to Say:**
  > *"For Part 5, we benchmarked our system. Our baseline unindexed linear scan took ~17.0 ms per query. By replacing it with a pre-computed sparse vector index and memoized state graphs, latency dropped to **0.003 ms**—a **5,600x speedup** with an ultra-compact memory footprint of just 23 KB.*
  > 
  > ***One Key Challenge & Solution:***  
  > *Handling ambiguous updates like `DEMO_016` ('Confirm interview slot might already be finished, but I cannot confirm it') was challenging. A naive keyword search might prematurely mark it completed. By modeling message grouping as a state machine with an explicit `unclear` conflict state, we prevented premature closure and retained high priority.*
  > 
  > *All 4 structured JSON artifacts (`priority_output.json`, `related_message_groups.json`, `privacy_routing_output.json`, `benchmark_comparison_report.json`) are generated and ready.  
  > Thank you for reviewing my L2 submission!"*

---

## 📋 Pre-Recording Checklist
- [ ] Run `python pipeline.py` to ensure all 4 JSON files are fresh.
- [ ] Run `streamlit run app.py` on your local browser (`localhost:8501`).
- [ ] Set browser zoom to 90% or 100% so all cards are clearly visible.
- [ ] Keep Loom bubble in the bottom corner and test your microphone.
- [ ] Follow the 5-minute timer so you stay within the 5-minute maximum limit!
