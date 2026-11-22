"""Atlas Operations Intelligence Assistant - backend.

Responsibilities
  * FR-01/02  load + validate operational issue records (pandas)
  * FR-03/04  SOP ingestion (PDF/TXT -> chunks -> local embeddings -> FAISS) and
              grounded question answering with LangChain + DeepSeek
  * FR-05/06  deterministic pandas metrics (the numerical source of truth)
  * FR-07     question routing (data questions -> pandas, procedure questions -> RAG)
  * FR-08     fixed-format HTML report
  * FR-09     n8n webhook client

No secrets live in this file; everything is read from environment variables.
"""
from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:  # optional: .env support
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

REQUIRED_COLUMNS = [
    "issue_id", "reported_at", "region", "team", "issue_category",
    "priority", "status", "resolution_time_hours", "issue_description",
]
VALID_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]
VALID_REGIONS = ["North", "South", "East", "West"]
UNRESOLVED = ["Open", "In Progress"]
COMPLETED = ["Resolved", "Closed"]
REPORT_TITLE = "Atlas Operations Performance Report"
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


class AtlasError(Exception):
    """User-facing error with a readable message."""


# --------------------------------------------------------------------------- #
# FR-01 / FR-02: loading, validation, preparation
# --------------------------------------------------------------------------- #
def load_issues_file(path: str | os.PathLike) -> pd.DataFrame:
    """Read a CSV or Excel file as strings (validation happens later)."""
    if not path:
        raise AtlasError("No operational issue file was provided. Upload a CSV or Excel file.")
    p = Path(str(path))
    if not p.exists():
        raise AtlasError(f"File not found: {p.name}")
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p, dtype=str, keep_default_na=False)
        else:
            raise AtlasError(f"Unsupported file type '{ext}'. Please upload a .csv, .xlsx or .xls file.")
    except AtlasError:
        raise
    except Exception as exc:
        raise AtlasError(f"The file could not be read as a table: {exc}") from exc
    if df.empty:
        raise AtlasError("The uploaded file contains no rows.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


@dataclass
class ValidationResult:
    data: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)      # fatal: no analysis possible
    warnings: list[str] = field(default_factory=list)    # rows excluded / values ignored
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors and self.data is not None and len(self.data) > 0


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def _canonical(series: pd.Series, allowed: list[str]) -> pd.Series:
    """Case-insensitive match to the canonical spelling; unknown values -> NA."""
    lookup = {a.lower(): a for a in allowed}
    return _clean_text(series).str.lower().map(lookup)


def validate_issues(raw: pd.DataFrame) -> ValidationResult:
    """Validate and prepare issue records. Never fabricates numbers.

    Fatal errors: missing required columns, nothing valid left.
    Row-level problems are reported as warnings and the affected rows are excluded
    (duplicates keep the first occurrence so that one row == one issue).
    """
    res = ValidationResult()
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append("Missing required column(s): " + ", ".join(missing))
        return res

    n_raw = len(df)
    df = df[REQUIRED_COLUMNS].copy()
    for c in ["issue_id", "region", "team", "issue_category", "priority", "status", "issue_description"]:
        df[c] = _clean_text(df[c]).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # --- issue ids -------------------------------------------------------- #
    blank_id = df["issue_id"].isna()
    if blank_id.any():
        res.warnings.append(f"{int(blank_id.sum())} row(s) excluded: blank issue_id.")
        df = df[~blank_id]
    dup = df.duplicated(subset="issue_id", keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        ids = ", ".join(sorted(df.loc[dup, "issue_id"].unique())[:10])
        res.warnings.append(
            f"{n_dup} duplicate issue_id row(s) removed (first occurrence kept): {ids}"
        )
        df = df[~dup]

    # --- reported_at ------------------------------------------------------ #
    df["reported_at"] = pd.to_datetime(df["reported_at"].astype(str).str.strip(), errors="coerce")
    bad_date = df["reported_at"].isna()
    if bad_date.any():
        res.warnings.append(
            f"{int(bad_date.sum())} row(s) excluded: reported_at missing or not a valid date "
            f"({', '.join(df.loc[bad_date, 'issue_id'].head(5))}...)."
        )
        df = df[~bad_date]

    # --- status / priority / region -------------------------------------- #
    for col, allowed, label in [("status", VALID_STATUSES, "status"),
                                ("priority", VALID_PRIORITIES, "priority")]:
        canon = _canonical(df[col], allowed)
        bad = canon.isna()
        if bad.any():
            vals = ", ".join(sorted(map(str, df.loc[bad, col].unique()))[:5])
            res.warnings.append(f"{int(bad.sum())} row(s) excluded: invalid {label} value(s): {vals}.")
        df[col] = canon
        df = df[~bad]
    canon_r = _canonical(df["region"], VALID_REGIONS)
    bad_r = canon_r.isna()
    if bad_r.any():
        res.warnings.append(
            f"{int(bad_r.sum())} row(s) have an unrecognised region and are left out of region analysis only."
        )
    df["region"] = canon_r
    for c in ("team", "issue_category"):
        miss = df[c].isna()
        if miss.any():
            res.warnings.append(f"{int(miss.sum())} row(s) have a blank {c}; shown as 'Unspecified'.")
            df[c] = df[c].fillna("Unspecified")

    # --- resolution time -------------------------------------------------- #
    # Blank stays NaN (legitimate for unresolved); never replaced with 0.
    rt_raw = df["resolution_time_hours"].astype(str).str.strip()
    rt_blank = rt_raw.isin(["", "nan", "None", "NaN"])
    rt = pd.to_numeric(rt_raw.where(~rt_blank), errors="coerce")
    malformed = (~rt_blank) & rt.isna()
    negative = rt < 0
    if malformed.any() or negative.any():
        res.warnings.append(
            f"{int((malformed | negative).sum())} resolution_time_hours value(s) are non-numeric or negative "
            "and are ignored for calculations."
        )
        rt = rt.mask(malformed | negative)
    df["resolution_time_hours"] = rt
    done = df["status"].isin(COMPLETED)
    n_done_missing = int((done & rt.isna()).sum())
    if n_done_missing:
        res.warnings.append(
            f"{n_done_missing} Resolved/Closed issue(s) have no valid resolution time; "
            "they are excluded from the average (not treated as 0)."
        )
    n_unres_with_time = int((~done & rt.notna()).sum())
    if n_unres_with_time:
        res.warnings.append(
            f"{n_unres_with_time} unresolved issue(s) carry a resolution time; it is ignored for the average."
        )

    df = df.sort_values(["reported_at", "issue_id"]).reset_index(drop=True)
    if df.empty:
        res.errors.append("No valid issue records remain after validation.")
        return res

    res.data = df
    res.stats = {
        "rows_in_file": n_raw,
        "valid_rows": len(df),
        "rows_excluded": n_raw - len(df),
        "duplicates_removed": n_dup,
    }
    return res


def data_overview(df: pd.DataFrame) -> dict[str, Any]:
    """FR-01 initial assessment of a validated frame."""
    return {
        "records": len(df),
        "columns": list(df.columns),
        "reporting_period": period_text(df),
        "categories": sorted(df["issue_category"].unique()),
        "teams": sorted(df["team"].unique()),
        "priorities": [p for p in VALID_PRIORITIES if p in set(df["priority"])],
        "statuses": [s for s in VALID_STATUSES if s in set(df["status"])],
        "regions": sorted(df["region"].dropna().unique()),
        "missing_resolution_values": int(df["resolution_time_hours"].isna().sum()),
    }


def overview_markdown(raw: pd.DataFrame, res: ValidationResult) -> str:
    lines = []
    if res.errors:
        lines.append("### ❌ Validation failed")
        lines += [f"- {e}" for e in res.errors]
        return "\n".join(lines)
    o = data_overview(res.data)
    lines += [
        "### ✅ Operational issue data loaded",
        f"- **Rows in file:** {res.stats['rows_in_file']}  |  **Valid issues:** {o['records']}",
        f"- **Reporting period:** {o['reporting_period']}",
        f"- **Issue categories ({len(o['categories'])}):** {', '.join(o['categories'])}",
        f"- **Teams ({len(o['teams'])}):** {', '.join(o['teams'])}",
        f"- **Priorities:** {', '.join(o['priorities'])}",
        f"- **Statuses:** {', '.join(o['statuses'])}",
        f"- **Regions:** {', '.join(o['regions'])}",
        f"- **Blank resolution times:** {o['missing_resolution_values']} "
        "(expected for unresolved issues; kept as missing, not zero)",
    ]
    if res.warnings:
        lines.append("\n**⚠️ Data-quality notes**")
        lines += [f"- {w}" for w in res.warnings]
    else:
        lines.append("\nNo data-quality problems found.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# FR-05 / FR-06: pandas metrics (single source of numerical truth)
# --------------------------------------------------------------------------- #
def period_text(df: pd.DataFrame) -> str:
    return f"{df['reported_at'].min():%Y-%m-%d} to {df['reported_at'].max():%Y-%m-%d}"


def _top(series: pd.Series) -> tuple[str, int, list[str]]:
    counts = series.value_counts()
    top_n = int(counts.iloc[0])
    leaders = sorted(counts[counts == top_n].index.tolist())
    return leaders[0], top_n, leaders


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    status = df["status"].value_counts().reindex(VALID_STATUSES, fill_value=0)
    completed = df[df["status"].isin(COMPLETED)]
    valid_times = completed["resolution_time_hours"].dropna()
    avg = float(valid_times.mean()) if len(valid_times) else None

    cat_counts = df["issue_category"].value_counts()
    team_counts = df["team"].value_counts()
    cat_name, cat_n, cat_ties = _top(df["issue_category"])
    team_name, team_n, team_ties = _top(df["team"])

    monthly = df.groupby(df["reported_at"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)
    region = df["region"].dropna().value_counts()

    return {
        "period": period_text(df),
        "period_start": f"{df['reported_at'].min():%Y-%m-%d}",
        "period_end": f"{df['reported_at'].max():%Y-%m-%d}",
        "total": int(df["issue_id"].nunique()),
        "status": {k: int(v) for k, v in status.items()},
        "unresolved": int(status["Open"] + status["In Progress"]),
        "completed": int(status["Resolved"] + status["Closed"]),
        "avg_resolution_hours": avg,
        "avg_resolution_n": int(len(valid_times)),
        "top_category": {"name": cat_name, "count": cat_n, "ties": cat_ties},
        "top_team": {"name": team_name, "count": team_n, "ties": team_ties},
        "category_counts": {k: int(v) for k, v in cat_counts.items()},
        "team_counts": {k: int(v) for k, v in team_counts.items()},
        "monthly_counts": {k: int(v) for k, v in monthly.items()},
        "region_counts": {k: int(v) for k, v in region.items()},
        "priority_counts": {k: int(v) for k, v in df["priority"].value_counts().reindex(VALID_PRIORITIES, fill_value=0).items()},
    }


def _fmt_avg(m: dict) -> str:
    if m["avg_resolution_hours"] is None:
        return "unavailable (no Resolved/Closed issues with a valid resolution time)"
    return f"{m['avg_resolution_hours']:.2f} hours"


def _table(d: dict[str, int], a: str, b: str = "Issues") -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in d.items())
    return f"| {a} | {b} |\n|---|---|\n{rows}"


# --------------------------------------------------------------------------- #
# FR-07: question routing
# --------------------------------------------------------------------------- #
SOP_PATTERNS = [
    r"\bsop\b", r"\bprocedure", r"\bescalat", r"\bresponsib", r"\bowns?\b", r"\bowner",
    r"\bpolic", r"\bguideline", r"\brout(e|ing)\b", r"\bdefin", r"\bwhat must happen",
    r"\bresponse (target|time)", r"\bphone\b", r"\b24/?7\b", r"\bhotline", r"\bcall[- ]?cent",
    r"\bwho (handles|should|is)\b", r"\bhow (do|should|to) (we|i|you)\b", r"\bnotify\b",
    r"\bconfidential", r"\bwhat does the (document|sop)", r"\bwhich team (handles|owns|is responsible)",
]
DATA_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bnumber of\b", r"\baverage\b", r"\bmean\b",
    r"\bmost (frequent|common)", r"\bhighest\b", r"\blargest\b", r"\bmost issues\b",
    r"\bper month\b", r"\bmonth", r"\bregion", r"\bunresolved\b", r"\bstatus breakdown\b",
    r"\bresolution time\b", r"\bsummary\b", r"\bstatistic", r"\bpercent", r"\bwhich (team|category|region)\b",
]


def route_question(q: str) -> str:
    """Return 'sop' or 'data'. Procedure/policy wording wins ties because the SOP
    never contains numeric metrics, whereas metrics wording ('how many', 'average')
    is always answered from pandas."""
    t = q.lower()
    # explicit numeric-metric phrasing goes to pandas even if "team" appears
    strong_data = re.search(
        r"how many|average|mean |total|number of|highest number|most (frequent|common|issues)|"
        r"received the (most|highest)|count|unresolved|per month|by month|by region", t)
    strong_sop = re.search(
        r"\bsop\b|escalat|responsible for|\bowns?\b|\bowner|who (handles|should)|"
        r"24/?7|phone|procedure|guideline|response target|what must happen|define", t)
    if strong_sop and not (strong_data and not re.search(r"\bsop\b|escalat|24/?7|phone|define|responsible", t)):
        return "sop"
    if strong_data:
        return "data"
    sop_hits = sum(bool(re.search(p, t)) for p in SOP_PATTERNS)
    data_hits = sum(bool(re.search(p, t)) for p in DATA_PATTERNS)
    return "sop" if sop_hits >= data_hits and sop_hits > 0 else "data"


def answer_data_question(q: str, df: pd.DataFrame) -> str:
    """Deterministic, pandas-backed answers. The LLM never produces these numbers."""
    m = compute_metrics(df)
    t = q.lower()
    st = m["status"]
    footer = f"\n\n_Source: operational issue records ({m['period']}), calculated with pandas._"

    if re.search(r"average|mean|resolution time|how long", t):
        return (
            f"**Average resolution time for completed issues: {_fmt_avg(m)}**\n\n"
            f"Calculated from {m['avg_resolution_n']} Resolved/Closed issues that have a valid resolution time "
            f"(Resolved: {st['Resolved']}, Closed: {st['Closed']}). Unresolved issues (Open: {st['Open']}, "
            f"In Progress: {st['In Progress']}) are excluded because their resolution time is not yet known; "
            "counting blanks as zero would understate the true average." + footer)
    if re.search(r"month|trend|august|september|october|over time", t):
        return "**Issues reported per month**\n\n" + _table(m["monthly_counts"], "Month") + footer
    if re.search(r"region|geograph|north|south|east|west", t):
        top = max(m["region_counts"], key=m["region_counts"].get)
        return (f"**Region with the most issues: {top} ({m['region_counts'][top]})**\n\n"
                + _table(m["region_counts"], "Region") + footer)
    if re.search(r"categor|type of issue|recurring|most (frequent|common)", t) and not re.search(r"team", t):
        c = m["top_category"]
        tie = f" (tied with {', '.join(x for x in c['ties'] if x != c['name'])})" if len(c["ties"]) > 1 else ""
        return (f"**Most frequent issue category: {c['name']} — {c['count']} issues{tie}**\n\n"
                + _table(m["category_counts"], "Category") + footer)
    if re.search(r"team|workload|received", t):
        c = m["top_team"]
        tie = f" (tied with {', '.join(x for x in c['ties'] if x != c['name'])})" if len(c["ties"]) > 1 else ""
        return (f"**Team with the highest issue count: {c['name']} — {c['count']} issues{tie}**\n\n"
                + _table(m["team_counts"], "Team")
                + "\n\nThis reflects recorded workload (issue count), not efficiency." + footer)
    if re.search(r"unresolved|still open|outstanding|pending", t):
        return (f"**Unresolved issues: {m['unresolved']}** (Open {st['Open']} + In Progress {st['In Progress']})" + footer)
    if re.search(r"in progress", t):
        return f"**In Progress issues: {st['In Progress']}**" + footer
    if re.search(r"\bopen\b", t):
        return f"**Open issues: {st['Open']}**" + footer
    if re.search(r"resolved or closed|completed|\bresolved\b|\bclosed\b", t):
        if re.search(r"resolved or closed|completed", t) or ("resolved" in t and "closed" in t):
            return (f"**Completed issues: {m['completed']}** (Resolved {st['Resolved']} + Closed {st['Closed']})" + footer)
        k = "Resolved" if "resolved" in t else "Closed"
        return f"**{k} issues: {st[k]}**" + footer
    if re.search(r"priorit|critical", t):
        return "**Issues by assigned priority**\n\n" + _table(m["priority_counts"], "Priority") + footer
    if re.search(r"status", t):
        return ("**Status breakdown**\n\n" + _table(st, "Status")
                + f"\n\nUnresolved: **{m['unresolved']}** · Completed: **{m['completed']}**" + footer)
    if re.search(r"how many|total|number of|reported", t):
        return (f"**Total reported issues: {m['total']}** during {m['period']}.\n\n"
                + _table(st, "Status") + footer)
    return (f"**Operational summary** ({m['period']})\n\n- Total issues: **{m['total']}**\n"
            f"- Unresolved: **{m['unresolved']}**\n- Average resolution time: **{_fmt_avg(m)}**\n"
            f"- Most frequent category: **{m['top_category']['name']}**\n"
            f"- Team with highest issue count: **{m['top_team']['name']}**" + footer)


# --------------------------------------------------------------------------- #
# FR-03 / FR-04: SOP processing and RAG
# --------------------------------------------------------------------------- #
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

RAG_PROMPT = """You are the Atlas Operations assistant for Northstar Systems.
Answer the question using ONLY the SOP context below.
Rules:
- Explain documented procedures; never create new policies, numbers, contacts or deadlines.
- If the context does not contain the answer, reply exactly that this cannot be verified from the uploaded SOP, and say what the SOP does state if relevant.
- Do not give issue counts or metrics; those come from the issue records.
- Be concise. Mention the SOP section when you can.

SOP context:
{context}

Question: {question}

Answer:"""


def extract_document_text(path: str | os.PathLike) -> str:
    if not path:
        raise AtlasError("No SOP document was provided. Upload a PDF or TXT file.")
    p = Path(str(path))
    if not p.exists():
        raise AtlasError(f"File not found: {p.name}")
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader

            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        elif ext == ".txt":
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            raise AtlasError(f"Unsupported document type '{ext}'. Upload a PDF or TXT file.")
    except AtlasError:
        raise
    except Exception as exc:
        raise AtlasError(f"The document could not be read: {exc}") from exc
    if not text.strip():
        raise AtlasError("No text could be extracted from the document (is it a scanned image?).")
    return text


class SOPKnowledgeBase:
    """Text -> chunks -> local embeddings -> FAISS."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.store = None
        self.chunks: list[str] = []
        self.source = ""

    def build(self, path: str | os.PathLike) -> int:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = extract_document_text(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "])
        self.chunks = [c for c in splitter.split_text(text) if c.strip()]
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            self.store = FAISS.from_texts(self.chunks, emb)
        except ImportError as exc:
            raise AtlasError(f"RAG dependencies are missing ({exc}). Run: pip install -r requirements.txt") from exc
        except Exception as exc:
            raise AtlasError(f"Could not build the embedding index: {exc}") from exc
        self.source = Path(str(path)).name
        return len(self.chunks)

    @property
    def ready(self) -> bool:
        return self.store is not None

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        if not self.ready:
            raise AtlasError("The SOP has not been processed yet. Upload and process an SOP document first.")
        return [d.page_content for d in self.store.similarity_search(question, k=k)]


def _get_llm():
    key = (os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key or key.startswith("your_"):
        raise AtlasError("LLM_API_KEY is not configured. Copy .env.example to .env and add your key.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise AtlasError("langchain-openai is not installed. Run: pip install -r requirements.txt") from exc
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "llama-3.3-70b-versatile"),
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0, timeout=60, max_retries=1)


def answer_sop_question(q: str, kb: SOPKnowledgeBase | None) -> str:
    if kb is None or not kb.ready:
        raise AtlasError("Upload and process the operations SOP (PDF/TXT) to ask procedure questions.")
    passages = kb.retrieve(q)
    context = "\n\n---\n\n".join(passages)
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chain = ChatPromptTemplate.from_template(RAG_PROMPT) | _get_llm() | StrOutputParser()
        answer = chain.invoke({"context": context, "question": q})
    except AtlasError:
        raise
    except Exception as exc:
        raise AtlasError(f"The language model request failed: {exc}") from exc
    sources = "\n\n".join(f"> {p.strip()[:300]}{'…' if len(p) > 300 else ''}" for p in passages[:3])
    return f"{answer.strip()}\n\n<details><summary>Retrieved SOP passages</summary>\n\n{sources}\n\n</details>"


def answer_question(q: str, df: pd.DataFrame | None, kb: SOPKnowledgeBase | None) -> tuple[str, str]:
    """Route and answer. Returns (markdown_answer, route_label)."""
    q = (q or "").strip()
    if not q:
        raise AtlasError("Please type a question.")
    route = route_question(q)
    if route == "data":
        if df is None:
            raise AtlasError("Upload and validate the operational issue file before asking data questions.")
        return answer_data_question(q, df), "📊 pandas analysis"
    return answer_sop_question(q, kb), "📄 SOP retrieval (RAG)"


# --------------------------------------------------------------------------- #
# FR-08: fixed-format HTML report
# --------------------------------------------------------------------------- #
def business_findings(m: dict) -> list[str]:
    f = []
    c, t = m["top_category"], m["top_team"]
    share = 100 * c["count"] / m["total"]
    f.append(f"<strong>{html.escape(c['name'])}</strong> was the most frequently reported issue category "
             f"with {c['count']} of {m['total']} issues ({share:.1f}%).")
    tshare = 100 * t["count"] / m["total"]
    f.append(f"<strong>{html.escape(t['name'])}</strong> is associated with the highest recorded workload "
             f"({t['count']} issues, {tshare:.1f}%). This reflects issue volume, not efficiency.")
    f.append(f"{m['unresolved']} issues remain unresolved "
             f"({m['status']['Open']} Open, {m['status']['In Progress']} In Progress); "
             f"{m['completed']} are completed.")
    if m["avg_resolution_hours"] is not None:
        f.append(f"Completed issues took an average of {m['avg_resolution_hours']:.2f} hours to resolve "
                 f"(based on {m['avg_resolution_n']} Resolved/Closed issues; unresolved issues excluded).")
    else:
        f.append("Average resolution time is unavailable because no completed issue has a valid resolution time.")
    mo = m["monthly_counts"]
    if len(mo) > 1:
        peak = max(mo, key=mo.get)
        f.append(f"Monthly volume peaked in <strong>{peak}</strong> with {mo[peak]} issues "
                 f"({', '.join(f'{k}: {v}' for k, v in mo.items())}).")
    rg = m["region_counts"]
    if rg:
        rtop = max(rg, key=rg.get)
        f.append(f"<strong>{rtop}</strong> recorded the most issues by region ({rg[rtop]}).")
    return f


def generate_report_html(df: pd.DataFrame | None) -> str:
    """Independent of chat history: depends only on the uploaded, validated dataset."""
    if df is None or df.empty:
        raise AtlasError("No validated operational data is available. Upload and validate the issue file first.")
    m = compute_metrics(df)
    e = html.escape
    th = "padding:8px 12px;border:1px solid #d0d7e2;background:#eef2f8;color:#14284b;text-align:left;font-size:14px;"
    td = "padding:8px 12px;border:1px solid #d0d7e2;font-size:14px;color:#1d2a3f;"

    def rows(d: dict, bold: str | None = None) -> str:
        return "".join(
            f"<tr><td style='{td}{'font-weight:bold;' if k == bold else ''}'>{e(str(k))}</td>"
            f"<td style='{td}text-align:right;{'font-weight:bold;' if k == bold else ''}'>{v}</td></tr>"
            for k, v in d.items())

    def section(title: str) -> str:
        return (f"<h2 style='font-size:17px;color:#14284b;border-bottom:2px solid #2f6fdf;"
                f"padding-bottom:4px;margin:26px 0 10px;'>{title}</h2>")

    def table(head_a, head_b, body) -> str:
        return (f"<table style='border-collapse:collapse;width:100%;max-width:480px;'><tr>"
                f"<th style='{th}'>{head_a}</th><th style='{th}text-align:right;'>{head_b}</th></tr>{body}</table>")

    avg = (f"{m['avg_resolution_hours']:.2f} hours" if m["avg_resolution_hours"] is not None
           else "Unavailable (no completed issues with a valid resolution time)")
    findings = "".join(f"<li style='color:#1d2a3f;margin-bottom:6px;'>{x}</li>" for x in business_findings(m))
    kpi = lambda label, val: (f"<td style='padding:12px;border:1px solid #d0d7e2;background:#f7f9fc;"
                              f"text-align:center;'><div style='font-size:12px;color:#5a6b85;'>{label}</div>"
                              f"<div style='font-size:22px;font-weight:bold;color:#14284b;'>{val}</div></td>")
    tm = m["top_team"]; cat = m["top_category"]
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{REPORT_TITLE}</title></head>
<body style="margin:0;padding:0;background:#f0f3f8;font-family:Arial,Helvetica,sans-serif;color:#1d2a3f;">
<div style="max-width:720px;margin:0 auto;background:#ffffff;color:#1d2a3f;padding:28px 32px;">
<h1 style="margin:0;font-size:24px;color:#14284b;">{REPORT_TITLE}</h1>
<p style="color:#1d2a3f;margin:4px 0 0;color:#5a6b85;font-size:13px;">Northstar Systems &middot; Generated {datetime.now():%Y-%m-%d %H:%M}</p>

{section('Reporting Period')}
<p style="color:#1d2a3f;font-size:15px;margin:0;">{e(m['period_start'])} to {e(m['period_end'])}</p>

{section('Total Reported Issues')}
<table style="border-collapse:collapse;width:100%;"><tr>{kpi('Total issues', m['total'])}{kpi('Unresolved', m['unresolved'])}{kpi('Completed', m['completed'])}{kpi('Avg resolution (h)', f"{m['avg_resolution_hours']:.2f}" if m['avg_resolution_hours'] is not None else 'n/a')}</tr></table>

{section('Issue Status Summary')}
{table('Status', 'Issues', rows(m['status']))}
<p style="color:#1d2a3f;font-size:13px;color:#5a6b85;">Unresolved = Open + In Progress ({m['unresolved']}). Completed = Resolved + Closed ({m['completed']}).</p>

{section('Leading Issue Category')}
<p style="color:#1d2a3f;font-size:15px;margin:0 0 8px;"><strong>{e(cat['name'])}</strong> &mdash; {cat['count']} issues</p>
{table('Category', 'Issues', rows(m['category_counts'], cat['name']))}

{section('Team Workload')}
<p style="color:#1d2a3f;font-size:15px;margin:0 0 8px;"><strong>{e(tm['name'])}</strong> has the highest issue count ({tm['count']} issues).</p>
{table('Team', 'Issues', rows(m['team_counts'], tm['name']))}

{section('Average Resolution Time')}
<p style="color:#1d2a3f;font-size:15px;margin:0;"><strong>{avg}</strong></p>
<p style="color:#1d2a3f;font-size:13px;color:#5a6b85;">Mean of resolution_time_hours for Resolved and Closed issues with valid values ({m['avg_resolution_n']} issues). Unresolved issues are excluded.</p>

{section('Additional Analysis: Monthly Issue Counts')}
{table('Month', 'Issues', rows(m['monthly_counts']))}

{section('Additional Analysis: Region-Wise Issue Counts')}
{table('Region', 'Issues', rows(m['region_counts']))}

{section('Business Findings')}
<ul style="color:#1d2a3f;font-size:14px;line-height:1.5;padding-left:20px;margin:0;">{findings}</ul>

<p style="color:#1d2a3f;margin-top:30px;font-size:11px;color:#8a97ad;border-top:1px solid #d0d7e2;padding-top:10px;">
All figures are calculated from the uploaded issue records using pandas. Fictional data for demonstration.</p>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# FR-09: n8n webhook client
# --------------------------------------------------------------------------- #
def validate_email_request(recipient: str, html_report: str) -> str | None:
    """Return an error message, or None when the request looks valid."""
    if not (recipient or "").strip():
        return "Please enter a recipient email address."
    if not EMAIL_RE.match(recipient.strip()):
        return f"'{recipient.strip()}' is not a valid email address."
    if not (html_report or "").strip():
        return "The report is empty. Generate the report before sending."
    return None


def send_report_via_n8n(recipient: str, html_report: str, title: str = REPORT_TITLE) -> tuple[bool, str]:
    """POST JSON to the n8n webhook. Returns (success, message based on the real response)."""
    err = validate_email_request(recipient, html_report)
    if err:
        return False, err
    url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not url:
        return False, "N8N_WEBHOOK_URL is not configured. Add it to your .env file."
    try:
        import requests
    except ImportError:
        return False, "The 'requests' package is not installed."
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("N8N_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["X-Atlas-Secret"] = secret
    payload = {"recipient": recipient.strip(), "report_title": title, "html_report": html_report}
    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    except requests.exceptions.ConnectionError:
        return False, "Could not reach n8n. Is n8n running and is the workflow active/listening?"
    except requests.exceptions.Timeout:
        return False, "n8n did not respond within 30 seconds."
    except requests.exceptions.RequestException as exc:
        return False, f"Request to n8n failed: {exc}"
    try:
        body = r.json()
    except ValueError:
        body = {}
    msg = body.get("message") if isinstance(body, dict) else None
    ok = r.ok and (not isinstance(body, dict) or body.get("status", "success") == "success")
    if ok:
        return True, msg or f"Report sent to {recipient.strip()}."
    return False, msg or f"n8n returned HTTP {r.status_code}: {r.text[:200] or 'no details'}"
