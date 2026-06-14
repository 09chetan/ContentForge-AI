from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Optional RAG layer (Pinecone + sentence-transformers)
# Set RAG_ENABLED=false to skip; defaults to True when pinecone is present
# ---------------------------------------------------------------------------
import importlib as _importlib

def _rag_available() -> bool:
    try:
        _importlib.import_module("pinecone")
        _importlib.import_module("sentence_transformers")
        return True
    except ImportError:
        return False

RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() != "false" and _rag_available()

if RAG_ENABLED:
    from rag_layer import (  # type: ignore
        upsert_evidence,
        upsert_section,
        retrieve_for_section,
        format_citations,
        collection_stats,
        retrieve,
    )

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages → FactCheck → SEO → Reviewer → Metrics → Evaluator)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ---- Fact Check schemas ----
class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "unverifiable"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    source_url: Optional[str] = None


class FactCheckReport(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0, description="0=fabricated, 1=fully grounded")
    hallucination_risk: Literal["low", "medium", "high"]
    claims: List[ClaimVerification]
    summary: str


# ---- SEO schemas ----
class SEOReport(BaseModel):
    seo_title: str = Field(..., description="Optimised <60 char title")
    meta_description: str = Field(..., description="150-160 char meta description")
    primary_keyword: str
    secondary_keywords: List[str] = Field(..., max_length=8)
    seo_score: int = Field(..., ge=0, le=100)
    recommendations: List[str] = Field(..., max_length=6)


# ---- Reviewer schemas ----
class GrammarIssue(BaseModel):
    message: str
    category: str
    severity: Literal["minor", "moderate", "major"]


class ReviewReport(BaseModel):
    readability_score: float = Field(..., ge=0.0, le=100.0, description="Flesch Reading Ease")
    readability_grade: str
    grammar_issues: List[GrammarIssue] = Field(default_factory=list)
    grammar_score: int = Field(..., ge=0, le=100)
    content_quality_score: int = Field(..., ge=0, le=100)
    tone_consistency: Literal["consistent", "inconsistent", "mixed"]
    structure_score: int = Field(..., ge=0, le=100)
    overall_score: int = Field(..., ge=0, le=100)
    strengths: List[str] = Field(default_factory=list, max_length=5)
    improvements: List[str] = Field(default_factory=list, max_length=6)
    revised_title: Optional[str] = None


# ---- Analytics schemas ----
class GenerationMetrics(BaseModel):
    topic: str
    mode: str
    word_count: int
    section_count: int
    evidence_count: int
    rag_enabled: bool
    fact_check_score: Optional[float]
    hallucination_risk: Optional[str]
    seo_score: Optional[int]
    review_overall_score: Optional[int]
    readability_score: Optional[float]
    eval_score: Optional[float] = None
    eval_faithfulness: Optional[float] = None
    timestamp: str


# ---- Evaluation schemas ----
class EvaluationReport(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Are claims grounded in evidence?")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="Does content answer the topic?")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Is retrieved context used?")
    overall_eval_score: float = Field(..., ge=0.0, le=1.0)
    eval_summary: str


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str

    # RAG metadata
    rag_enabled: bool
    rag_stats: dict

    # Fact Check, SEO, Review, Metrics, Evaluation
    fact_check_report: Optional[dict]
    seo_report: Optional[dict]
    review_report: Optional[dict]
    generation_metrics: Optional[dict]
    eval_report: Optional[dict]


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4.1-mini")

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 4b) RAG Index Node
# Runs after research; embeds evidence into Pinecone
# -----------------------------
def rag_index_node(state: State) -> dict:
    """Embed research evidence into the vector store."""
    if not RAG_ENABLED:
        return {"rag_enabled": False, "rag_stats": {}}

    evidence = state.get("evidence", []) or []
    topic = state.get("topic", "")

    count = upsert_evidence(evidence, topic=topic)
    stats = collection_stats()

    return {
        "rag_enabled": True,
        "rag_stats": {"indexed_evidence": count, **stats},
    }


# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
                "rag_enabled": state.get("rag_enabled", False),
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    # ---- RAG: retrieve semantically relevant context ----
    rag_context = ""
    rag_citations: List[dict] = []
    if RAG_ENABLED and payload.get("rag_enabled", False):
        rag_context = retrieve_for_section(
            task_title=task.title,
            bullets=task.bullets,
            topic=payload["topic"],
            top_k=8,
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                    + (f"\n\n{rag_context}" if rag_context else "")
                )
            ),
        ]
    ).content.strip()

    # ---- RAG: index the written section for downstream sections ----
    if RAG_ENABLED and payload.get("rag_enabled", False):
        upsert_section(
            section_md=section_md,
            task_id=task.id,
            task_title=task.title,
            topic=payload["topic"],
        )

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


# -----------------------------
# 9) Fact Check Node
# -----------------------------
FACT_CHECK_SYSTEM = """You are a rigorous fact-checking agent for technical blog posts.

Extract up to 12 specific, verifiable claims (statistics, product names, dates, API names,
model names, benchmarks). For each claim:
- verdict: "supported" if backed by provided evidence URLs, "unsupported" if contradicted,
  "unverifiable" if no evidence either way.
- confidence: your confidence in the verdict (0.0–1.0).
- source_url: matching evidence URL if applicable, else null.

overall_score: fraction of supported claims out of total.
hallucination_risk: "low" if score>0.8, "medium" if 0.5–0.8, "high" if <0.5.
"""

def fact_check_node(state: State) -> dict:
    checker = llm.with_structured_output(FactCheckReport)
    merged = state.get("merged_md") or state.get("final") or ""
    if not merged:
        return {"fact_check_report": None}

    evidence = state.get("evidence", []) or []
    evidence_text = "\n".join(
        f"- {e.title if hasattr(e,'title') else e.get('title','')} | "
        f"{e.url if hasattr(e,'url') else e.get('url','')} | "
        f"{(e.snippet if hasattr(e,'snippet') else e.get('snippet',''))[:120]}"
        for e in evidence[:20]
    )

    report = checker.invoke([
        SystemMessage(content=FACT_CHECK_SYSTEM),
        HumanMessage(content=(
            f"Evidence available:\n{evidence_text or 'None (closed_book mode)'}\n\n"
            f"Blog content:\n{merged[:6000]}"
        )),
    ])
    return {"fact_check_report": report.model_dump()}


# -----------------------------
# 10) SEO Node
# -----------------------------
SEO_SYSTEM = """You are an expert technical SEO strategist.

Analyse a technical blog post and produce:
- seo_title: compelling, keyword-rich, under 60 chars.
- meta_description: 150-160 chars summarising the post for search snippets.
- primary_keyword: the single most important keyword phrase.
- secondary_keywords: 4-8 supporting keyword phrases.
- seo_score: 0-100 score based on: keyword placement in title/headings (30pts),
  content depth (25pts), readability (20pts), internal structure (15pts),
  meta quality (10pts).
- recommendations: up to 6 concrete, actionable SEO improvements.
"""

def seo_node(state: State) -> dict:
    analyser = llm.with_structured_output(SEOReport)
    merged = state.get("merged_md") or state.get("final") or ""
    if not merged:
        return {"seo_report": None}

    plan = state.get("plan")
    blog_title = ""
    if hasattr(plan, "blog_title"):
        blog_title = plan.blog_title
    elif isinstance(plan, dict):
        blog_title = plan.get("blog_title", "")

    report = analyser.invoke([
        SystemMessage(content=SEO_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {blog_title}\n"
            f"Topic: {state['topic']}\n\n"
            f"Content:\n{merged[:6000]}"
        )),
    ])
    return {"seo_report": report.model_dump()}


# -----------------------------
# 11) Reviewer Node
# -----------------------------
REVIEWER_SYSTEM = """You are a senior editorial reviewer for technical blog posts.

Evaluate the blog on:
- Grammar and language quality (grammar_score 0-100)
- Content quality: depth, accuracy, originality, examples (content_quality_score 0-100)
- Tone consistency across sections (consistent/inconsistent/mixed)
- Structure: intro, body flow, conclusion, headings (structure_score 0-100)
- overall_score: weighted average (grammar 20%, content 40%, tone 15%, structure 25%)

readability_score and readability_grade are computed separately — leave as 0.0 and "N/A",
they will be filled in programmatically.

List up to 5 strengths and 6 concrete improvements.
Optionally suggest a revised_title if the current one is weak.
"""

def reviewer_node(state: State) -> dict:
    import textstat  # type: ignore

    merged = state.get("merged_md") or state.get("final") or ""
    if not merged:
        return {"review_report": None}

    # Programmatic readability (no LLM needed)
    plain_text = re.sub(r"[#*`\[\]()>]", "", merged)
    flesch = textstat.flesch_reading_ease(plain_text)
    grade = textstat.text_standard(plain_text, float_output=False)

    reviewer = llm.with_structured_output(ReviewReport)
    plan = state.get("plan")
    blog_title = plan.blog_title if hasattr(plan, "blog_title") else (plan or {}).get("blog_title", "")

    report = reviewer.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {blog_title}\n"
            f"Audience: {plan.audience if hasattr(plan,'audience') else ''}\n"
            f"Tone: {plan.tone if hasattr(plan,'tone') else ''}\n\n"
            f"Content:\n{merged[:7000]}"
        )),
    ])

    report_dict = report.model_dump()
    report_dict["readability_score"] = round(flesch, 1)
    report_dict["readability_grade"] = grade

    return {"review_report": report_dict}


# -----------------------------
# 12) Metrics Node
# -----------------------------
def metrics_node(state: State) -> dict:
    from datetime import datetime

    merged = state.get("merged_md") or state.get("final") or ""
    word_count = len(merged.split())
    sections = state.get("sections") or []
    evidence = state.get("evidence") or []

    fc = state.get("fact_check_report") or {}
    seo = state.get("seo_report") or {}
    review = state.get("review_report") or {}

    metrics = GenerationMetrics(
        topic=state.get("topic", ""),
        mode=state.get("mode", ""),
        word_count=word_count,
        section_count=len(sections),
        evidence_count=len(evidence),
        rag_enabled=state.get("rag_enabled", False),
        fact_check_score=fc.get("overall_score"),
        hallucination_risk=fc.get("hallucination_risk"),
        seo_score=seo.get("seo_score"),
        review_overall_score=review.get("overall_score"),
        readability_score=review.get("readability_score"),
        eval_score=None,  # filled after eval_node runs
        eval_faithfulness=None,  # filled after eval_node runs
        timestamp=datetime.utcnow().isoformat(),
    )

    # Append to persistent JSONL log
    log_path = Path("analytics_log.jsonl")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(metrics.model_dump_json() + "\n")

    return {"generation_metrics": metrics.model_dump()}


# -----------------------------
# 13) Evaluation Node (NEW)
# -----------------------------
EVAL_SYSTEM = """You are an LLM evaluation judge scoring a generated blog post.

Score each dimension from 0.0 to 1.0:

- faithfulness: Are all specific claims (stats, names, dates) grounded in the provided evidence? 
  1.0 = everything cited, 0.0 = all fabricated.
- answer_relevance: Does the full blog directly and thoroughly answer the original topic?
  1.0 = perfectly on-topic, 0.0 = completely off-topic.
- context_recall: Was the retrieved evidence/context actually used in the writing?
  1.0 = all context used, 0.0 = context ignored.
- overall_eval_score: weighted average (faithfulness 40%, answer_relevance 40%, context_recall 20%).
- eval_summary: 2-3 sentence plain English verdict.

Be strict. A score above 0.85 must be genuinely excellent.
"""

def eval_node(state: State) -> dict:
    evaluator = llm.with_structured_output(EvaluationReport)
    merged = state.get("merged_md") or state.get("final") or ""
    if not merged:
        return {"eval_report": None}

    evidence = state.get("evidence", []) or []
    evidence_text = "\n".join(
        f"- {e.title if hasattr(e,'title') else e.get('title','')} | "
        f"{e.url if hasattr(e,'url') else e.get('url','')}"
        for e in evidence[:15]
    )

    rag_context = ""
    if state.get("rag_enabled") and RAG_ENABLED:
        try:
            hits = retrieve(state.get("topic", ""), top_k=5)
            rag_context = "\n".join(h.get("chunk", "")[:150] for h in hits)
        except Exception:
            rag_context = "Error retrieving RAG context"

    report = evaluator.invoke([
        SystemMessage(content=EVAL_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state.get('topic', '')}\n"
            f"Mode: {state.get('mode', '')}\n\n"
            f"Evidence provided:\n{evidence_text or 'None'}\n\n"
            f"RAG context used:\n{rag_context or 'None'}\n\n"
            f"Generated blog (first 5000 chars):\n{merged[:5000]}"
        )),
    ])
    return {"eval_report": report.model_dump()}


# -----------------------------
# 14) PDF Ingestion Function (standalone, called from frontend)
# -----------------------------
def ingest_pdf_to_rag(pdf_path: str, topic: str = "knowledge_base") -> dict:
    """
    Extract text from a PDF, chunk it, and upsert into the RAG vector store.
    Returns ingestion summary.
    """
    if not RAG_ENABLED:
        return {"error": "RAG not enabled"}

    try:
        import fitz  # pymupdf
    except ImportError:
        return {"error": "pymupdf not installed. Run: pip install pymupdf"}

    doc = fitz.open(pdf_path)
    full_text = "\n\n".join(page.get_text() for page in doc)
    doc.close()

    if not full_text.strip():
        return {"error": "No extractable text found in PDF"}

    from rag_layer import _chunk_text, embed, _stable_id, _get_client
    
    filename = Path(pdf_path).name
    chunks = _chunk_text(full_text, max_tokens=300)
    vectors = embed(chunks)
    
    # Pinecone upsert
    index = _get_client()
    index.upsert(vectors=[
        {
            "id": _stable_id(f"pdf:{filename}:{i}:{c[:80]}"),
            "values": v,
            "metadata": {
                "type": "pdf_knowledge",
                "topic": topic,
                "source": filename,
                "chunk": c,
                "page_approx": i,
            }
        }
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ])

    return {
        "file": filename,
        "chunks_indexed": len(chunks),
        "total_chars": len(full_text),
    }


# -----------------------------
# 15) Build reducer subgraph
# -----------------------------
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 16) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("rag_index", rag_index_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)
g.add_node("fact_check", fact_check_node)
g.add_node("seo", seo_node)
g.add_node("reviewer", reviewer_node)
g.add_node("metrics", metrics_node)
g.add_node("evaluator", eval_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "rag_index")
g.add_edge("rag_index", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", "fact_check")
g.add_edge("fact_check", "seo")
g.add_edge("seo", "reviewer")
g.add_edge("reviewer", "metrics")
g.add_edge("metrics", "evaluator")
g.add_edge("evaluator", END)

app = g.compile()
app