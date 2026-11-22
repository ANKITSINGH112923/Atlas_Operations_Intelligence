"""Atlas Operations Intelligence Assistant - Gradio UI.  Run: python app.py"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

import backend as be

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def _path(f):
    return getattr(f, "name", f) if f else None


def process_issues(file, state):
    try:
        raw = be.load_issues_file(_path(file))
    except be.AtlasError as e:
        state["df"] = None
        return f"### ❌ {e}", state
    res = be.validate_issues(raw)
    state["df"] = res.data if res.ok else None
    return be.overview_markdown(raw, res), state


def process_sop(file, state):
    kb = be.SOPKnowledgeBase()
    try:
        n = kb.build(_path(file))
    except be.AtlasError as e:
        state["kb"] = None
        return f"### ❌ {e}", state
    state["kb"] = kb
    return f"### ✅ SOP processed\n{kb.source}: {n} chunks embedded locally and stored in FAISS.", state


def load_samples(state):
    a, state = process_issues(str(SAMPLE_DIR / "operations_issues.csv"), state)
    b, state = process_sop(str(SAMPLE_DIR / "operations_sop.pdf"), state)
    return a, b, state


def ask(question, state):
    try:
        ans, route = be.answer_question(question, state.get("df"), state.get("kb"))
        return ans, route
    except be.AtlasError as e:
        return f"⚠️ {e}", "—"


def make_report(state):
    try:
        h = be.generate_report_html(state.get("df"))
    except be.AtlasError as e:
        state["report"] = ""
        return "", f"❌ {e}", state
    state["report"] = h
    return h, "✅ Report generated from the current dataset.", state


def send_email(recipient, state):
    ok, msg = be.send_report_via_n8n(recipient, state.get("report", ""))
    return ("✅ " if ok else "❌ ") + msg


with gr.Blocks(title="Atlas Operations Intelligence Assistant") as demo:
    state = gr.State({"df": None, "kb": None, "report": ""})
    gr.Markdown("# 🧭 Atlas Operations Intelligence Assistant\n"
                "Northstar Systems · Upload issue records and the SOP, ask questions, generate and email reports.")
    with gr.Tab("1 · Upload data"):
        with gr.Row():
            with gr.Column():
                issues_file = gr.File(label="Operational issue records (CSV / Excel)",
                                      file_types=[".csv", ".xlsx", ".xls"])
                issues_btn = gr.Button("Load & validate issues", variant="primary")
                issues_out = gr.Markdown()
            with gr.Column():
                sop_file = gr.File(label="Operations SOP (PDF / TXT)", file_types=[".pdf", ".txt"])
                sop_btn = gr.Button("Process SOP", variant="primary")
                sop_out = gr.Markdown()
        sample_btn = gr.Button("Load bundled sample data")
    with gr.Tab("2 · Ask Atlas"):
        q = gr.Textbox(label="Question", lines=2,
                       placeholder="e.g. How many issues are unresolved? / What must happen for Critical issues?")
        gr.Examples([
            "How many operational issues were reported?", "How many issues are Open, In Progress, Resolved and Closed?",
            "Which issue category occurred most frequently?", "Which team received the highest number of issues?",
            "What is the average resolution time for completed issues?", "How did issue counts change by month?",
            "Which region had the most issues?", "Who owns a Data Quality issue according to the SOP?",
            "What must happen for Critical-priority issues?", "Does the SOP provide a 24/7 customer support phone number?",
        ], inputs=q)
        ask_btn = gr.Button("Ask", variant="primary")
        route_out = gr.Textbox(label="Answered by", interactive=False)
        answer_out = gr.Markdown(label="Answer")
    with gr.Tab("3 · Report & email"):
        rep_btn = gr.Button("Generate Atlas Operations Performance Report", variant="primary")
        rep_status = gr.Markdown()
        rep_html = gr.HTML()
        email = gr.Textbox(label="Recipient email", placeholder="manager@example.com")
        send_btn = gr.Button("Send Email")
        send_out = gr.Markdown()

    issues_btn.click(process_issues, [issues_file, state], [issues_out, state])
    sop_btn.click(process_sop, [sop_file, state], [sop_out, state])
    sample_btn.click(load_samples, [state], [issues_out, sop_out, state])
    ask_btn.click(ask, [q, state], [answer_out, route_out])
    q.submit(ask, [q, state], [answer_out, route_out])
    rep_btn.click(make_report, [state], [rep_html, rep_status, state])
    send_btn.click(send_email, [email, state], [send_out])

if __name__ == "__main__":
    demo.launch()
