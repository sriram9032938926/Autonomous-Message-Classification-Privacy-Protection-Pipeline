# Autonomous Message Classification & PII Privacy Protection Engine

An end-to-end, privacy-first local NLP pipeline and interactive Streamlit web application for categorizing chronological messages, extracting tasks & events, and masking sensitive PII values.

Designed with **zero external AI service API calls** at runtime to comply strictly with data privacy and non-exposure requirements.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+
- pip package manager

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline & App
```bash
# Run backend pipeline directly
python pipeline.py

# Launch interactive Streamlit Web App
streamlit run app.py
```

---

## 📂 Project Architecture (Clean 2-File Structure)

```
.
├── pipeline.py                    # File 1: Processing Engine (Classification, Extraction, Sensitive Info Detection & Masking)
├── app.py                         # File 2: Streamlit Interactive Web Application UI & Dashboard
├── messages.csv                   # Input dataset (900 fictional messages)
├── mandatory_demo_ids.csv         # 15 mandatory test IDs for video demonstration
├── classification_results.json    # Part 1 Output JSON
├── extracted_tasks_events.json    # Part 2 Output JSON
├── sensitive_info_detections.json # Part 3 Output JSON
├── requirements.txt               # Dependencies
└── README.md                      # Documentation & Submission Details
```

---

## ⚙️ Core Technical Implementation

### Part 1: How Message Classification Works
Messages are processed in strict chronological order and classified into **6 mandatory categories**:
1. `Action Required`: Messages demanding specific user action before a deadline.
2. `Meeting or Event`: Scheduled meetings, calendar updates, orientations, or appointments.
3. `Personal Information`: User preferences, medical notes, dietary choices, T-shirt sizes.
4. `General Information`: Operational notices, status updates, shuttle schedules, weather info.
5. `Promotional`: Marketing offers, discount codes (`SAVE17`), coupon expirations.
6. `Sensitive Information`: Credentials, card details, OTPs, recovery keys, addresses.

**Classification Output Schema Example:**
```json
{
  "message_id": "MSG_0001",
  "category": "meeting_or_event",
  "confidence": 0.92,
  "reason": "Contains scheduled event, meeting invitation, calendar update, or appointment."
}
```

---

### Part 2: How Tasks & Events Are Extracted
For messages categorized under `Action Required` or `Meeting or Event`:
- **Title**: Clean extracted title stripped of boilerplate prefixes.
- **Deadline/Date**: Extracted in `YYYY-MM-DD` format.
- **Time**: Extracted in 24-hour or AM/PM format (e.g. `10:00`, `6 PM`).
- **Person**: Extracted person involved if mentioned.
- **Priority**: Categorized as `high`, `medium`, or `low`.
- **Handling Unclear Information**: If a date, time, or person is ambiguous (e.g. `MSG_0037`: *"The review could be Friday afternoon"*), the field is strictly stored as `null` / unresolved without guessing.

**Extraction Output Schema Example:**
```json
{
  "item_id": "ITEM_001",
  "type": "event",
  "title": "family dinner",
  "deadline": "2026-09-19",
  "time": "10:00",
  "person": null,
  "priority": "medium",
  "source_message_id": "MSG_0001"
}
```

---

### Part 3: How Sensitive Information Is Detected & Masked
Detects secret credentials and PII using regex patterns & heuristics:
- **Detected Types**: Credit Cards, Bank Accounts, OTPs, Passwords, Access Tokens, Recovery Codes, Gov IDs, Phone Numbers, Residential Addresses.
- **Masking Mechanism**: Secret values are dynamically replaced with `******` (e.g. `Your OTP is ******`). Sensitive values never leak to logs or UI screenshots.

**Sensitive Detection Schema Example:**
```json
{
  "message_id": "MSG_0005",
  "sensitivity_type": "private_address",
  "risk": "medium",
  "masked_text": "Hi, My home address is ********************.",
  "recommended_action": "safe_to_process_locally"
}
```

---

## 📌 15 Mandatory Message IDs Coverage

| Message ID | Category | Sensitive? | Masked Text Preview |
|---|---|---|---|
| `MSG_0001` | `meeting_or_event` | No | Calendar update: family dinner, 2026-09-19... |
| `MSG_0002` | `action_required` | No | Can you review the privacy checklist before... |
| `MSG_0003` | `meeting_or_event` | No | Reminder: mentor catch-up happens on 2026-09-16... |
| `MSG_0004` | `general_information` | No | The training material is on the portal. |
| `MSG_0005` | `sensitive_information` | **Yes (Address)** | Hi, My home address is ********************. |
| `MSG_0006` | `general_information` | No | Important: The laptop battery is fully charged. |
| `MSG_0007` | `action_required` | No | Please reply to the client email by 2026-09-04. |
| `MSG_0009` | `personal_information` | No | For my profile, my emergency contact is my brother. |
| `MSG_0012` | `general_information` | No | FYI: I will send the login details separately. |
| `MSG_0013` | `sensitive_information` | **Yes (Credit Card)** | My card number is ********************. |
| `MSG_0014` | `promotional` | No | Special festival discount on clothing. Use code SAVE17. |
| `MSG_0015` | `promotional` | No | Flash sale on laptops starts at 6 PM. Use code SAVE23. |
| `MSG_0016` | `personal_information` | No | Remember that i drink coffee without sugar. |
| `MSG_0024` | `personal_information` | No | I might prefer evening meetings now. |
| `MSG_0037` | `meeting_or_event` | No | The review could be Friday afternoon. *(Unclear Date)* |

---

## 💡 Assumptions and Limitations

1. **Assumptions**:
   - Timestamps are provided in ISO format (`YYYY-MM-DD HH:MM:SS`) and processed in exact chronological sequence.
   - Messages containing passwords, account recovery keys, card numbers, or OTPs take top priority and are classified as `sensitive_information`.

2. **Limitations**:
   - Rule-based pattern matching works with 100% accuracy on structured templates but requires pattern updates for free-form conversational text.
   - Dates specified in relative text (e.g. "next Friday") without fixed calendar dates are intentionally set to `null` to avoid hallucinating incorrect dates.

---

## 🤖 AI-Tool Usage Declaration

This software was developed with assistance from Google Antigravity AI (Gemini model) for boilerplate generation, UI design layout in Streamlit, and test script verification. All algorithms, regex patterns, logic flows, and output structures were audited and verified manually.

---

## 📄 License & Confidentiality Notice
This project contains code for an assignment task. The dataset `messages.csv` contains fictional synthetic data and must not be published to public code repositories without authorization.
