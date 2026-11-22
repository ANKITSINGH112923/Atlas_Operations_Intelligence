# Atlas Operations Intelligence Assistant

Local AI assistant for the fictional Northstar Systems Operations Team. It combines:

- **pandas** for every number (issue counts, statuses, category/team/region/month analysis, average resolution time)
- **RAG** (pypdf → LangChain splitter → local HuggingFace embeddings → FAISS → DeepSeek via LangChain) for SOP questions
- **Gradio** UI with question routing, a fixed-format **HTML report**, and **n8n** webhook email delivery

It is an analytics/reporting assistant, not a ticketing or forecasting system.

## 15-Minute Video Explanation
<!-- TODO: paste your video link here -->

## Architecture
```
 CSV/Excel ──► validate_issues ──► pandas metrics ──┐
                                                    ├─► Router ─► Gradio answer
 PDF/TXT ──► chunks ─► local embeddings ─► FAISS ───┘            (data → pandas, procedure → RAG+DeepSeek)

 Gradio "Send Email" ─► POST JSON ─► n8n Webhook ─► IF valid? ─► Send Email ─► Respond 200 {status:"success"}
                                                       └─ no ──► Respond 400 {status:"error"}
```
Files: `app.py` (UI), `backend.py` (all logic), `sample_data/`, `.env.example`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then set DEEPSEEK_API_KEY and N8N_WEBHOOK_URL
python app.py               # opens http://127.0.0.1:7860
```
The first SOP run downloads the `all-MiniLM-L6-v2` embedding model (needs internet once; then runs locally).
In the app click **Load bundled sample data** (or upload `sample_data/operations_issues.csv` and `operations_sop.pdf`).

## n8n workflow (local)
Start n8n (`npx n8n`), create a workflow and **activate** it. Use the **Production URL** in `N8N_WEBHOOK_URL`.

1. **Webhook** – Method `POST`, Path `atlas-report`, Respond: *Using 'Respond to Webhook' Node*.
2. **IF** – Combine: AND. Conditions (on `{{ $json.body.… }}`):
   - `{{ $json.body.recipient }}` *matches regex* `^[^\s@]+@[^\s@]+\.[^\s@]+$`
   - `{{ $json.body.html_report }}` *is not empty*
   - (optional) `{{ $json.headers['x-atlas-secret'] }}` *equals* your `N8N_WEBHOOK_SECRET`
3. **Send Email** (true branch) – SMTP credentials configured inside n8n (never in this repo).
   To: `{{ $json.body.recipient }}` · Subject: `{{ $json.body.report_title }}` · Email Format: **HTML** · HTML: `{{ $json.body.html_report }}`
4. **Respond to Webhook** (after Send Email) – Code `200`, body JSON: `{"status":"success","message":"Report sent to {{ $('Webhook').item.json.body.recipient }}."}`
5. **Respond to Webhook** (false branch) – Code `400`, body JSON: `{"status":"error","message":"Invalid request: recipient email or report content is missing/invalid."}`

Payload sent by Atlas: `{"recipient": "...", "report_title": "Atlas Operations Performance Report", "html_report": "<html>…"}`.
Atlas also checks the recipient/report locally before posting, so n8n's error branch mainly protects against other callers; to demo the n8n error branch, call it with `curl -X POST <url> -H "Content-Type: application/json" -d '{"recipient":"bad","report_title":"t","html_report":""}'`.

## Verified results for the sample data (pandas)
| Metric | Value |
|---|---|
| Total issues | 84 (period 2026-08-02 to 2026-10-31) |
| Open / In Progress / Resolved / Closed | 5 / 7 / 60 / 12 (unresolved 12) |
| Most frequent category | Access & Login (24) |
| Team with highest issue count | IT Support (34) |
| Avg resolution time | 22.23 h (72 completed issues) |
| Monthly (Aug/Sep/Oct) | 25 / 30 / 29 |
| Regions N/W/E/S | 28 / 19 / 19 / 18 |

## Example questions
- *Data (pandas):* How many issues are unresolved? Which team received the most issues? What is the average resolution time? How did counts change by month?
- *SOP (RAG):* What must happen for Critical-priority issues? Who owns a Data Quality issue? Does the SOP provide a 24/7 support phone number? (→ cannot be verified; SOP §8 says it does not.)

## Validation behaviour
Missing columns, unreadable/unsupported files and empty data produce a clear error. Duplicate `issue_id`s are removed (first kept), rows with bad dates/statuses/priorities are excluded and listed, non-numeric/negative resolution times are ignored, blank resolution times stay missing (never 0), and the average uses only Resolved/Closed rows with valid values (reported as *unavailable* if none).

## Testing
Manual test matrix (FR-10): SOP routing/priority answer · Critical escalation · 24/7 number unknown · total count · status counts · category + team · average · monthly/region · valid email · invalid email · empty report · bad/missing data. Add screenshots to `screenshots/` (`01_app_interface.png` … `08_error_handling.png`).

## Security
No secrets in code. Keep `.env` out of git (`.gitignore`). `.env.example` holds placeholders only.

## Known limitations
- Question routing is rule-based (keywords); unusual phrasing may route to the wrong component.
- Data answers are templated (not LLM-written) by design, so they can't be invented.
- Retrieval quality depends on the PDF text extraction and the small local embedding model.
- No authentication on the Gradio app; intended for local use.
