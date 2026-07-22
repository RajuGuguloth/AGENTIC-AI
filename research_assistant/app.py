import gradio as gr
import asyncio
from typing import List, Tuple
from fastapi import FastAPI
from config import Config
from etl.pdf_loader import load_pdfs_from_files
try:
    from etl.arxiv_loader import search_and_load
except ImportError:
    def search_and_load(*args, **kwargs):
        raise NotImplementedError("arxiv_loader not available — use Deep-Read with arXiv URL instead.")
from etl.chunker import chunk_documents_parent_child
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.multimodal_retriever import MultimodalRetriever
from orchestration.document_chat import chat_turn, _append_turn
from orchestration.deep_read.orchestrator import run_deep_read
from orchestration.simple_summarizer import summarize_from_index, format_llm_error
from observability.feedback import append_feedback
from api.feedback_router import router as feedback_router
from memory.user_memory import resolve_user_id, start_session, update_from_feedback
import uuid

# ── Global State ──────────────────────────────────────────────────────────────
INDEX_STORE_PATH = "./index_store/"

DEFAULT_SUMMARY_GOAL = (
    "Summarize the uploaded PDF in simple, clear language. Include:\n"
    "- What the document is about (main topic)\n"
    "- Key points and findings\n"
    "- Important conclusions or recommendations\n"
    "Use short sections and bullet points. Avoid jargon."
)

dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
hybrid_retriever = MultimodalRetriever(dense_retriever, sparse_retriever)

if hybrid_retriever.load(INDEX_STORE_PATH):
    print(f"[app] Restored index from {INDEX_STORE_PATH}")
else:
    print(f"[app] No persisted index found at {INDEX_STORE_PATH}")


# ── Indexing Functions ────────────────────────────────────────────────────────
def build_index_from_pdfs(files, progress=gr.Progress()) -> str:
    """Load PDFs from uploaded files, chunk, and build Hybrid index."""
    if not files:
        return "No files provided."
    try:
        file_paths = [f.name for f in files]
        progress(0.1, desc="Loading PDFs...")
        raw_docs = load_pdfs_from_files(file_paths, progress=progress)
        
        progress(0.3, desc="Chunking documents (Parent-Child + Semantic)...")
        parent_docs, child_docs = chunk_documents_parent_child(raw_docs, progress=progress)
        
        progress(0.7, desc="Building Dense Index...")
        dense_retriever.build_index(child_docs)
        
        progress(0.8, desc="Building Sparse Index (BM25)...")
        sparse_retriever.build_index(child_docs)
        
        progress(0.9, desc="Storing Parent Documents...")
        hybrid_retriever.add_parents(parent_docs)

        progress(0.95, desc="Saving index to disk...")
        hybrid_retriever.save(INDEX_STORE_PATH)
        
        progress(1.0, desc="Indexing Complete!")
        return (
            f"✅ Index ready ({len(child_docs)} chunks). "
            f"Go to **🔍 Research** tab and click **Summarize PDF Now**, "
            f"or use **Build Index & Summarize PDF** below for one-click summary."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error indexing PDFs: {e}"


def build_index_from_arxiv(query: str, max_results: int, progress=gr.Progress()) -> str:
    """Fetch arXiv papers, chunk, and build Hybrid index."""
    try:
        progress(0.1, desc="Fetching arXiv papers...")
        raw_docs = search_and_load(query, max_results=max_results, progress=progress)
        
        progress(0.3, desc="Chunking documents (Parent-Child + Semantic)...")
        parent_docs, child_docs = chunk_documents_parent_child(raw_docs, progress=progress)
        
        progress(0.7, desc="Building Dense Index...")
        dense_retriever.build_index(child_docs)
        
        progress(0.8, desc="Building Sparse Index (BM25)...")
        sparse_retriever.build_index(child_docs)
        
        progress(0.9, desc="Storing Parent Documents...")
        hybrid_retriever.add_parents(parent_docs)
        
        progress(1.0, desc="Indexing Complete!")
        return f"Successfully indexed {len(child_docs)} child chunks from {len(raw_docs)} arXiv papers."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error indexing arXiv: {e}"


# ── Research Orchestration ────────────────────────────────────────────────────
async def run_simple_summary_workflow(goal: str) -> Tuple[str, str, str, str]:
    """Fast path: retrieve + one LLM call (best for PDF summarize)."""
    if not hybrid_retriever.dense.is_ready:
        return (
            "Error: Index not built. Upload a PDF under **📚 Data Sources** and click **Build Index** first.",
            "",
            "",
            "",
        )

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return f"Configuration Error: {e}", "", "", ""

    try:
        result = await summarize_from_index(goal, llm, hybrid_retriever)
        chart_html = (
            "<div style='padding: 12px; background: #f0fdf4; border-radius: 8px;'>"
            "Simple summary mode — no chart generated."
            "</div>"
        )
        return (
            result["report"],
            chart_html,
            result["sub_queries"],
            result["findings"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return format_llm_error(e), "", "", ""


async def run_research_workflow(goal: str, max_queries: int) -> Tuple[str, str, str, str]:
    """
    Execute the full Agentic RAG workflow:
    Plan -> Parallel Research -> Synthesize -> Visualize
    """
    if not hybrid_retriever.dense.is_ready:
        return "Error: Index not built. Please build the index first from the Data sources tab.", "", "", ""

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return f"Configuration Error: {e}", "", "", ""

    try:
        from orchestration.planner import ResearchPlanner
        from orchestration.agent import run_parallel_research
        from orchestration.synthesizer import ReportSynthesizer
        from visualization.chart_generator import ChartGenerator

        # 1. Plan
        planner = ResearchPlanner(llm)
        sub_queries = await planner.plan(goal, max_queries=max_queries)
        sub_queries_md = "### Generated Sub-Queries\n" + "\n".join([f"- {q}" for q in sub_queries])

        # 2. Parallel Research
        results = await run_parallel_research(sub_queries, llm, hybrid_retriever)
        
        # Format intermediate findings
        findings_md = "### Intermediate Findings\n"
        for res in results:
            findings_md += f"**Q: {res['sub_query']}**\n\n{res['answer']}\n\n---\n\n"

        # 3. Synthesize
        synthesizer = ReportSynthesizer(llm)
        final_report = await synthesizer.synthesize(goal, results)

        # 4. Visualize
        chart_gen = ChartGenerator(llm)
        has_chart, img_b64, chart_msg = await chart_gen.extract_and_plot(final_report)
        
        chart_html = ""
        if has_chart and img_b64:
            chart_html = f"<div><h3>Auto-Generated Visualization</h3><p>{chart_msg}</p><img src='data:image/png;base64,{img_b64}' style='max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'/></div>"
        else:
            chart_html = f"<div style='padding: 20px; background: #f3f4f6; border-radius: 8px;'><i>{chart_msg}</i></div>"

        return final_report, chart_html, sub_queries_md, findings_md

    except Exception as e:
        import traceback
        traceback.print_exc()
        return format_llm_error(e), "", "", ""


def launch_research(goal: str, max_queries: int) -> Tuple[str, str, str, str]:
    """Gradio wrapper for full agentic workflow."""
    return asyncio.run(run_research_workflow(goal, max_queries))


def launch_simple_summary(goal: str = None) -> Tuple[str, str, str, str]:
    """Gradio wrapper for simple PDF summary."""
    return asyncio.run(run_simple_summary_workflow(goal or DEFAULT_SUMMARY_GOAL))


def summarize_indexed_pdf(max_queries: int = 3) -> Tuple[str, str, str, str]:
    """Run the default PDF summary workflow against the current index."""
    return launch_simple_summary(DEFAULT_SUMMARY_GOAL)


def build_index_and_summarize(files, max_queries: int = 3, progress=gr.Progress()):
    """Index uploaded PDFs, then immediately generate a summary."""
    status = build_index_from_pdfs(files, progress=progress)
    if not hybrid_retriever.dense.is_ready:
        return status, "", "", "", ""

    progress(1.0, desc="Generating summary...")
    report, chart, sub_queries, findings = launch_simple_summary()
    return (
        f"{status}\n\n📄 Summary generated — see **🔍 Research → 📝 Final Report**.",
        report,
        chart,
        sub_queries,
        findings,
    )


# ── Document Chat (NotebookLM-style) ─────────────────────────────────────────
async def _run_chat_turn(message: str, history: list, session_id: str) -> tuple:
    if not hybrid_retriever.dense.is_ready:
        reply = "⚠️ No documents indexed yet. Upload a PDF under **📚 Data Sources** and click **Build Index** first."
        return "", _append_turn(history, message, reply), "", session_id

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return "", _append_turn(history, message, f"Configuration error: {e}"), "", session_id

    sid = session_id or str(uuid.uuid4())
    user_id = resolve_user_id(session_id=sid)
    if Config.LEVEL5_ENABLED:
        start_session(user_id, sid)

    updated_history, sources, trace_id = await chat_turn(
        message, history or [], llm, hybrid_retriever, user_id=user_id
    )
    return "", updated_history, sources, trace_id, sid


def respond_to_chat(message: str, history: list, session_id: str) -> tuple:
    """Gradio wrapper for one chat turn."""
    if not message or not message.strip():
        return message, history or [], "", "", session_id or ""
    return asyncio.run(_run_chat_turn(message, history or [], session_id or ""))


def submit_chat_feedback(trace_id: str, rating: str, session_id: str) -> str:
    """Record thumbs up/down for the last chat answer (also available via POST /api/feedback)."""
    if not (trace_id or "").strip():
        return "Send a message first — no trace id yet."
    try:
        user_id = resolve_user_id(session_id=session_id or "")
        append_feedback(
            trace_id.strip(),
            rating,
            metadata={"path": "chat", "source": "gradio", "user_id": user_id},
        )
        if Config.LEVEL5_ENABLED:
            update_from_feedback(user_id, rating)
        label = "👍 helpful" if rating == "positive" else "👎 not helpful"
        return f"Thanks — feedback recorded ({label}). Trace: `{trace_id[:8]}...`"
    except Exception as exc:
        return f"Could not save feedback: {exc}"


def clear_chat(session_id: str) -> tuple:
    return [], "", "", "", str(uuid.uuid4())


# ── Deep-Read (structured paper analysis) ─────────────────────────────────────
async def _run_deep_read_async(paper_url: str, pdf_file, progress=gr.Progress()):
    progress(0, desc="Starting Deep-Read…")
    try:
        Config.validate()
    except Exception as e:
        return f"Configuration error: {e}", None, None, ""

    def prog_cb(fraction: float, msg: str):
        progress(fraction, desc=msg)

    try:
        if pdf_file is not None:
            path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
            result = await run_deep_read(path, progress=prog_cb, is_upload=True)
        elif paper_url and paper_url.strip():
            result = await run_deep_read(paper_url.strip(), progress=prog_cb, is_upload=False)
        else:
            return "Provide a paper URL (arXiv/PDF link) or upload a PDF.", None, None, ""

        status = (
            f"✅ **Deep-Read complete** ({result.latency_sec:.0f}s)\n\n"
            f"- **Title:** {result.title}\n"
            f"- **Pages:** {result.page_count} · **Figures:** {result.figure_count} · **Tables:** {result.table_count}\n"
            f"- **Job ID:** `{result.job_id}`\n"
            f"- **Artifacts:** `{result.artifacts_dir}`"
        )
        return status, result.report_markdown, result.report_path, result.ppt_path
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return format_llm_error(exc), None, None, ""


def run_deep_read_ui(paper_url: str, pdf_file, progress=gr.Progress()):
    return asyncio.run(_run_deep_read_async(paper_url, pdf_file, progress=progress))


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def create_ui():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(theme=theme, title="Agentic RAG Research Assistant") as demo:
        gr.Markdown(
            """
            # 🧠 Agentic RAG Research Assistant
            **Hybrid Search (FAISS + BM25 + RRF) | LangChain | Parallel Sub-queries**
            """
        )

        with gr.Tabs():
            # Tab 1: Research Workflow
            with gr.TabItem("🔍 Research"):
                index_ready = "✅ Index loaded — ready to summarize" if hybrid_retriever.dense.is_ready else "⚠️ No index yet — upload a PDF under **📚 Data Sources** first"
                gr.Markdown(
                    f"""
                    ### Get your PDF summary here
                    After indexing, click **Summarize PDF Now**. Your summary appears in **📝 Final Report** below.

                    **Status:** {index_ready}
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        goal_input = gr.Textbox(
                            label="Research Goal (or use default summary prompt below)",
                            value=DEFAULT_SUMMARY_GOAL,
                            lines=5
                        )
                    with gr.Column(scale=1):
                        max_queries_slider = gr.Slider(
                            minimum=1, maximum=6, value=3, step=1,
                            label="Max Sub-queries"
                        )
                        summarize_btn = gr.Button("📄 Summarize PDF Now", variant="primary")
                        research_btn = gr.Button("🚀 Run Agentic RAG")

                with gr.Tabs():
                    with gr.TabItem("📝 Final Report"):
                        report_output = gr.Markdown()
                    with gr.TabItem("📊 Visualizations"):
                        chart_output = gr.HTML()
                    with gr.TabItem("🧠 Agent Thinking"):
                        sub_queries_output = gr.Markdown()
                        findings_output = gr.Markdown()

                research_btn.click(
                    fn=launch_research,
                    inputs=[goal_input, max_queries_slider],
                    outputs=[report_output, chart_output, sub_queries_output, findings_output]
                )
                summarize_btn.click(
                    fn=summarize_indexed_pdf,
                    inputs=[max_queries_slider],
                    outputs=[report_output, chart_output, sub_queries_output, findings_output]
                )

            # Tab 2: Document Chat
            with gr.TabItem("💬 Chat with Documents"):
                chat_index_ready = (
                    "✅ Documents indexed — ask anything about your PDF"
                    if hybrid_retriever.dense.is_ready
                    else "⚠️ Index a PDF first under **📚 Data Sources**"
                )
                gr.Markdown(
                    f"""
                    ### NotebookLM-style chat
                    Ask follow-up questions about your uploaded documents. Each answer is grounded in your indexed PDF.

                    **Status:** {chat_index_ready}

                    **Try asking:**
                    - What is this document about?
                    - What are the main conclusions?
                    - Explain the key technical terms in simple language
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Document Chat", height=420)
                        chat_input = gr.Textbox(
                            label="Your question",
                            placeholder="Ask anything about your indexed PDF...",
                            lines=2,
                        )
                        with gr.Row():
                            chat_send_btn = gr.Button("Send", variant="primary")
                            chat_clear_btn = gr.Button("Clear chat")
                    with gr.Column(scale=1):
                        chat_sources = gr.Markdown(label="Sources for last answer")
                        chat_trace_id = gr.Textbox(
                            label="Trace ID (for feedback API)",
                            interactive=False,
                            placeholder="Appears after each answer",
                        )
                        feedback_status = gr.Markdown("")
                        with gr.Row():
                            feedback_up = gr.Button("👍 Helpful")
                            feedback_down = gr.Button("👎 Not helpful")

                chat_session_id = gr.State(str(uuid.uuid4()))
                chat_msg = [chat_input, chatbot, chat_session_id]
                chat_out = [chat_input, chatbot, chat_sources, chat_trace_id, chat_session_id]

                chat_send_btn.click(fn=respond_to_chat, inputs=chat_msg, outputs=chat_out)
                chat_input.submit(fn=respond_to_chat, inputs=chat_msg, outputs=chat_out)
                chat_clear_btn.click(
                    fn=clear_chat,
                    inputs=[chat_session_id],
                    outputs=[chatbot, chat_input, chat_sources, chat_trace_id, chat_session_id, feedback_status],
                )

                feedback_up.click(
                    fn=lambda tid, sid: submit_chat_feedback(tid, "positive", sid),
                    inputs=[chat_trace_id, chat_session_id],
                    outputs=[feedback_status],
                )
                feedback_down.click(
                    fn=lambda tid, sid: submit_chat_feedback(tid, "negative", sid),
                    inputs=[chat_trace_id, chat_session_id],
                    outputs=[feedback_status],
                )

            # Tab 3: Deep-Read
            with gr.TabItem("📖 Deep-Read"):
                gr.Markdown(
                    """
                    ### Structured paper analysis (how researchers actually read)
                    Paste an **arXiv link**, **PDF URL**, or **upload a PDF**. Deep-Read will:
                    1. Download & extract text, **figures**, and **tables**
                    2. Analyze section-by-section: metadata → problem → method → figures → results → limitations
                    3. Merge into a **detailed report** + **PowerPoint** with paper images

                    *Written for first-time readers — deep but understandable.*
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        deep_read_url = gr.Textbox(
                            label="Paper URL",
                            placeholder="https://arxiv.org/abs/1706.03762 or direct PDF link",
                            lines=1,
                        )
                        deep_read_file = gr.File(
                            label="Or upload PDF",
                            file_types=[".pdf"],
                            file_count="single",
                        )
                        deep_read_btn = gr.Button("🔬 Run Deep-Read", variant="primary")
                        deep_read_status = gr.Markdown("")
                    with gr.Column(scale=3):
                        deep_read_report = gr.Markdown(label="Structured Report")

                with gr.Row():
                    deep_read_md_dl = gr.File(label="Download Report (.md)", interactive=False)
                    deep_read_ppt_dl = gr.File(label="Download Slides (.pptx)", interactive=False)

                deep_read_btn.click(
                    fn=run_deep_read_ui,
                    inputs=[deep_read_url, deep_read_file],
                    outputs=[deep_read_status, deep_read_report, deep_read_md_dl, deep_read_ppt_dl],
                )

            # Tab 4: Data Sources (Indexing)
            with gr.TabItem("📚 Data Sources"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### Index Local PDFs
                            **Option A (easiest):** Upload PDF → **Build Index & Summarize PDF** — one click, summary appears in **🔍 Research → 📝 Final Report**.

                            **Option B:** Upload PDF → **Build Hybrid Index from PDFs** → go to **🔍 Research** → **Summarize PDF Now**.
                            """
                        )
                        pdf_file_input = gr.File(
                            label="Upload PDFs",
                            file_count="multiple",
                            file_types=[".pdf"]
                        )
                        index_and_summarize_btn = gr.Button(
                            "Build Index & Summarize PDF",
                            variant="primary",
                        )
                        index_pdf_btn = gr.Button("Build Hybrid Index from PDFs (index only)")
                        pdf_status = gr.Textbox(label="Status", interactive=False)

                        index_pdf_btn.click(
                            fn=build_index_from_pdfs,
                            inputs=[pdf_file_input],
                            outputs=[pdf_status]
                        )

                        index_and_summarize_btn.click(
                            fn=build_index_and_summarize,
                            inputs=[pdf_file_input, max_queries_slider],
                            outputs=[pdf_status, report_output, chart_output, sub_queries_output, findings_output],
                        )
                        
                    with gr.Column():
                        gr.Markdown("### Fetch from arXiv")
                        arxiv_query_input = gr.Textbox(
                            label="arXiv Search Query",
                            value="cat:cs.AI and LLM",
                            placeholder="e.g., all:transformer"
                        )
                        arxiv_max_slider = gr.Slider(
                            minimum=5, maximum=50, value=20, step=5,
                            label="Max Papers to Fetch"
                        )
                        index_arxiv_btn = gr.Button("Fetch & Build Hybrid Index")
                        arxiv_status = gr.Textbox(label="Status", interactive=False)
                        
                        index_arxiv_btn.click(
                            fn=build_index_from_arxiv,
                            inputs=[arxiv_query_input, arxiv_max_slider],
                            outputs=[arxiv_status]
                        )
                        
            # Tab 5: Configuration
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown(
                    f"""
                    ### Current Settings (from `.env`)
                    - **LLM Backend**: {Config.LLM_BACKEND}
                    - **LLM Provider (auto)**: {Config.LLM_PROVIDER}
                    - **Gemini Model**: {Config.GEMINI_MODEL}
                    - **OpenAI Model**: {Config.OPENAI_MODEL}
                    - **Embedding Model**: {Config.EMBEDDING_MODEL}
                    - **Perplexity Model**: {Config.PERPLEXITY_MODEL}
                    - **Chunk Size**: {Config.CHUNK_SIZE} chars (Overlap: {Config.CHUNK_OVERLAP})
                    - **Hybrid RRF K**: {Config.RRF_K}

                    **BYOK:** Set `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `LLM_PROVIDER=openai|gemini|auto`

                    *Note: Change settings in `.env` and restart the app.*
                    """
                )

    return demo


def create_app():
    """FastAPI app with Gradio UI and /api/feedback endpoint."""
    demo = create_ui()
    api = FastAPI(
        title="Agentic RAG Research Assistant",
        description="Hybrid RAG with Gemini verification and feedback API",
    )
    api.include_router(feedback_router)
    return gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    application = create_app()
    uvicorn.run(application, host="0.0.0.0", port=7860)
