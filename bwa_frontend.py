from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app, ingest_pdf_to_rag


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    # Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "image_specs": [],     # optional (not persisted)
                    "final": md_text,      # markdown body
                    "fact_check_report": None,
                    "seo_report": None,
                    "review_report": None,
                    "eval_report": None,
                    "generation_metrics": None,
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout - Updated with all 12 tabs including Evaluation
tab_plan, tab_evidence, tab_rag, tab_preview, tab_factcheck, tab_seo, tab_review, tab_eval, tab_kb, tab_analytics, tab_images, tab_logs = st.tabs([
    "🧩 Plan", "🔎 Evidence", "🧠 RAG", "📝 Preview",
    "✅ Fact Check", "📈 SEO", "📋 Review", "🧪 Evaluation",
    "📚 Knowledge Base", "📊 Analytics", "🖼️ Images", "🧾 Logs"
])

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "rag_enabled": False,
        "rag_stats": {},
        "fact_check_report": None,
        "seo_report": None,
        "review_report": None,
        "eval_report": None,
        "generation_metrics": None,
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- RAG tab ---
    with tab_rag:
        st.subheader("🧠 RAG / Vector Store")
        rag_enabled = out.get("rag_enabled", False)
        rag_stats = out.get("rag_stats") or {}

        if not rag_enabled:
            st.info(
                "RAG is disabled. Install `pinecone` and `sentence-transformers` to enable.\n\n"
                "```\npip install pinecone sentence-transformers\n```"
            )
        else:
            st.success("✅ RAG is active — evidence was embedded into Pinecone.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Indexed Evidence Chunks", rag_stats.get("indexed_evidence", "—"))
            col2.metric("Total Vectors in DB", rag_stats.get("vectors_count", "—"))
            col3.metric("Namespaces", len(rag_stats.get("namespaces", [])))

            with st.expander("Full RAG stats"):
                st.json(rag_stats)

            st.divider()
            st.caption(
                "Sections were written with semantically retrieved context from the vector store. "
                "Each worker section also got indexed back so later sections could reference earlier ones."
            )

            # Live query box
            st.subheader("🔍 Query the Vector Store")
            rag_query = st.text_input("Semantic search query", placeholder="e.g. transformer attention mechanism")
            if rag_query.strip():
                try:
                    from rag_layer import retrieve  # type: ignore
                    hits = retrieve(rag_query.strip(), top_k=6)
                    if hits:
                        for h in hits:
                            with st.expander(f"[{h.get('type','?')}] score={h.get('score','?')} — {h.get('title') or h.get('task_title','chunk')}"):
                                st.write(h.get("snippet") or h.get("chunk", ""))
                                if h.get("url"):
                                    st.markdown(f"[Source]({h['url']})")
                    else:
                        st.info("No results found.")
                except ImportError:
                    st.warning("rag_layer not available.")

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Fact Check tab ---
    with tab_factcheck:
        st.subheader("✅ Fact Check & Hallucination Report")
        fc = out.get("fact_check_report")
        if not fc:
            st.info("Fact check not available (no content generated yet).")
        else:
            risk = fc.get("hallucination_risk", "unknown")
            score = fc.get("overall_score", 0)
            color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")

            col1, col2 = st.columns(2)
            col1.metric("Factual Grounding Score", f"{score:.0%}")
            col2.metric("Hallucination Risk", f"{color} {risk.upper()}")

            st.info(fc.get("summary", ""))

            claims = fc.get("claims", [])
            if claims:
                df = pd.DataFrame([
                    {
                        "Claim": c["claim"][:80],
                        "Verdict": c["verdict"],
                        "Confidence": f"{c['confidence']:.0%}",
                        "Source": c.get("source_url") or "—",
                    }
                    for c in claims
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Raw fact check JSON"):
                st.json(fc)

    # --- SEO tab ---
    with tab_seo:
        st.subheader("📈 SEO Analysis")
        seo = out.get("seo_report")
        if not seo:
            st.info("SEO report not available yet.")
        else:
            score = seo.get("seo_score", 0)
            col1, col2 = st.columns([1, 3])
            col1.metric("SEO Score", f"{score}/100")

            with col2:
                st.progress(score / 100)

            st.markdown(f"**SEO Title:** `{seo.get('seo_title', '')}`")
            st.markdown(f"**Meta Description:** {seo.get('meta_description', '')}")
            st.markdown(f"**Primary Keyword:** `{seo.get('primary_keyword', '')}`")

            sec_kw = seo.get("secondary_keywords", [])
            if sec_kw:
                st.markdown("**Secondary Keywords:** " + " · ".join(f"`{k}`" for k in sec_kw))

            recs = seo.get("recommendations", [])
            if recs:
                st.markdown("**Recommendations:**")
                for r in recs:
                    st.markdown(f"- {r}")

            with st.expander("Raw SEO JSON"):
                st.json(seo)

    # --- Review tab ---
    with tab_review:
        st.subheader("📋 Editorial Review")
        rv = out.get("review_report")
        if not rv:
            st.info("No review report yet.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", f"{rv.get('overall_score', 0)}/100")
            col2.metric("Grammar", f"{rv.get('grammar_score', 0)}/100")
            col3.metric("Content Quality", f"{rv.get('content_quality_score', 0)}/100")
            col4.metric("Structure", f"{rv.get('structure_score', 0)}/100")

            st.metric("Readability (Flesch)", rv.get("readability_score", "—"))
            st.caption(f"Grade level: {rv.get('readability_grade', '—')} · Tone: {rv.get('tone_consistency', '—')}")

            if rv.get("revised_title"):
                st.info(f"💡 Suggested title: **{rv['revised_title']}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**✅ Strengths**")
                for s in rv.get("strengths", []):
                    st.markdown(f"- {s}")
            with col_b:
                st.markdown("**🔧 Improvements**")
                for i in rv.get("improvements", []):
                    st.markdown(f"- {i}")

            issues = rv.get("grammar_issues", [])
            if issues:
                with st.expander(f"Grammar issues ({len(issues)})"):
                    st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)

    # --- Evaluation tab (NEW) ---
    with tab_eval:
        st.subheader("🧪 LLM Evaluation (RAGAS-style)")
        ev = out.get("eval_report")
        if not ev:
            st.info("No evaluation report yet.")
        else:
            overall = ev.get("overall_eval_score", 0)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", f"{overall:.2f} / 1.0")
            col2.metric("Faithfulness", f"{ev.get('faithfulness', 0):.2f}")
            col3.metric("Answer Relevance", f"{ev.get('answer_relevance', 0):.2f}")
            col4.metric("Context Recall", f"{ev.get('context_recall', 0):.2f}")

            # Visual score bar
            st.progress(overall)

            grade = (
                "🟢 Excellent" if overall >= 0.85 else
                "🟡 Good" if overall >= 0.65 else
                "🔴 Needs Improvement"
            )
            st.markdown(f"**Grade: {grade}**")
            st.info(ev.get("eval_summary", ""))

            with st.expander("Raw evaluation JSON"):
                st.json(ev)

    # --- Knowledge Base tab ---
    with tab_kb:
        st.subheader("📚 PDF Knowledge Base")
        st.caption("Upload research papers or documents to enrich the RAG vector store.")

        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        kb_topic = st.text_input("Topic tag for this document", value="knowledge_base")

        if uploaded_pdf and st.button("📥 Ingest into Knowledge Base"):
            import tempfile
            from bwa_backend import ingest_pdf_to_rag

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            with st.spinner("Extracting and indexing PDF..."):
                result = ingest_pdf_to_rag(tmp_path, topic=kb_topic)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success(
                    f"✅ Indexed **{result['chunks_indexed']} chunks** "
                    f"from `{result['file']}` ({result['total_chars']:,} chars)"
                )

    # --- Analytics tab ---
    with tab_analytics:
        st.subheader("📊 Analytics Dashboard")

        log_path = Path("analytics_log.jsonl")
        if not log_path.exists():
            st.info("No analytics data yet. Generate a blog to start tracking.")
        else:
            records = []
            with log_path.open() as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass

            if not records:
                st.info("Log file exists but has no valid records yet.")
            else:
                df = pd.DataFrame(records)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Blogs Generated", len(df))
                col2.metric("Avg Word Count", f"{df['word_count'].mean():.0f}")
                col3.metric("Avg SEO Score", f"{df['seo_score'].dropna().mean():.0f}" if "seo_score" in df else "—")
                col4.metric("Avg Review Score", f"{df['review_overall_score'].dropna().mean():.0f}" if "review_overall_score" in df else "—")

                st.divider()

                # Score trends table
                display_cols = [c for c in [
                    "timestamp", "topic", "mode", "word_count",
                    "seo_score", "review_overall_score", "readability_score",
                    "fact_check_score", "hallucination_risk", "rag_enabled"
                ] if c in df.columns]

                st.dataframe(df[display_cols].sort_values("timestamp", ascending=False),
                             use_container_width=True, hide_index=True)

                # Charts
                if "seo_score" in df.columns and df["seo_score"].notna().any():
                    st.line_chart(df[["seo_score", "review_overall_score", "readability_score"]].dropna())

                # Download full log
                st.download_button(
                    "⬇️ Download Analytics Log",
                    data=log_path.read_text(),
                    file_name="analytics_log.jsonl",
                    mime="application/jsonl",
                )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")