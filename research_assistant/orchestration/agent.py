"""
Orchestration — Parallel Agent with Self-RAG
Executes research independently for each sub-query.
Implements Corrective RAG (Relevance Grading) and Hallucination Grading.
"""

import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from retrieval.hybrid_retriever import HybridRetriever
from config import Config

try:
    from verification.gemini_verifier import GeminiVerifier
    HAS_GEMINI_VERIFIER = True
except ImportError:
    HAS_GEMINI_VERIFIER = False


class RelevanceScore(BaseModel):
    is_relevant: bool = Field(description="True if the context contains information relevant to answering the query.")


class HallucinationScore(BaseModel):
    is_grounded: bool = Field(description="True if the answer is completely grounded in the context.")


class SubQueryAgent:
    """Answers a single sub-query with Self-RAG mechanisms."""

    def __init__(self, llm):
        self.llm = llm
        self._gemini_verifier = None
        if Config.LLM_BACKEND == "gemini" and HAS_GEMINI_VERIFIER:
            try:
                self._gemini_verifier = GeminiVerifier.from_config()
            except Exception as exc:
                print(f"[agent] GeminiVerifier unavailable: {exc}")

        self.relevance_grader = self.llm.with_structured_output(RelevanceScore)
        self.hallucination_grader = self.llm.with_structured_output(HallucinationScore)
        
        # Relevance Prompt
        self.relevance_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user query.
If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant.
Document:
{context}

Query: {query}""",
            input_variables=["context", "query"]
        )

        # Hallucination Prompt
        self.hallucination_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts.
Context:
{context}

Answer:
{answer}""",
            input_variables=["context", "answer"]
        )

        # Generation Prompt (Updated with Strict Guardrails)
        self.gen_prompt = PromptTemplate(
            template="""You are an expert Research AI Assistant operating within an advanced Hybrid-RAG pipeline. 
Your task is to answer the user's research query using ONLY the provided context blocks.

[CRITICAL PIPELINE GUARDRAILS]
1. STRICT TRUTHFULNESS: Analyze the provided contexts carefully. If the context does not contain enough information to answer the query, state explicitly: "I cannot find sufficient evidence in the current indexed literature to answer this." Do NOT use your pre-trained general knowledge to fill in gaps.
2. CITATION REQUIREMENT: Every factual claim, finding, or metric you output MUST be appended with its corresponding source index identifier from the context (e.g., "[Paper ID / Author Year]").
3. CONFLICT RESOLUTION: If different retrieved papers contradict each other, synthesize the contradiction clearly. Do not arbitrate; present the landscape of data neutrally.
4. AGENTIC ANALYSIS: If the query requires multi-step reasoning, break down your output into:
   - Synthesis of Evidence
   - Direct Answer to Query
   - Identified Gaps in Context

[RETRIEVED CONTEXT]
{context}

[USER RESEARCH QUERY]
{sub_query}

[THOUGHTFUL SCIENTIFIC RESPONSE]""",
            input_variables=["sub_query", "context"]
        )

        # Fallback Query Rewriter Prompt
        self.rewrite_prompt = PromptTemplate(
            template="""You are a query re-writer. The following query failed to retrieve relevant documents.
Look at the query and rewrite it to be broader and more optimized for a vector database search.
Original Query: {query}

Rewritten Query:""",
            input_variables=["query"]
        )

    async def grade_relevance(self, query: str, docs: List[Document]) -> List[Document]:
        """Filter out irrelevant documents."""
        relevant_docs = []
        for doc in docs:
            if self._gemini_verifier:
                try:
                    cosine = doc.metadata.get("retrieval_score")
                    if await self._gemini_verifier.is_relevant(
                        query,
                        doc.page_content,
                        score=cosine,
                        score_is_cosine=cosine is not None,
                    ):
                        relevant_docs.append(doc)
                except Exception as e:
                    print(f"[agent] Gemini relevance grading failed (fail-closed): {e}")
                continue

            chain = self.relevance_prompt | self.relevance_grader
            try:
                res = await chain.ainvoke({"query": query, "context": doc.page_content})
                if res.is_relevant:
                    relevant_docs.append(doc)
            except Exception as e:
                print(f"[agent] Relevance grading failed (fail-closed): {e}")
        return relevant_docs

    async def rewrite_query(self, query: str) -> str:
        """Rewrite query for fallback."""
        chain = self.rewrite_prompt | self.llm
        res = await chain.ainvoke({"query": query})
        return res.content if hasattr(res, 'content') else str(res)

    async def check_hallucination(self, answer: str, context: str) -> bool:
        """Check if answer is grounded."""
        if self._gemini_verifier:
            try:
                return await self._gemini_verifier.is_grounded(answer, context)
            except Exception as e:
                print(f"[agent] Gemini groundedness check failed (fail-closed): {e}")
                return False

        chain = self.hallucination_prompt | self.hallucination_grader
        try:
            res = await chain.ainvoke({"answer": answer, "context": context})
            return res.is_grounded
        except Exception as e:
            print(f"[agent] Hallucination grading failed (fail-closed): {e}")
            return False

    async def answer(self, sub_query: str, context: str) -> str:
        """Generate an answer for the sub-query."""
        if not context.strip():
            return "I cannot find sufficient evidence in the current indexed literature to answer this."
            
        chain = self.gen_prompt | self.llm
        response = await chain.ainvoke({"sub_query": sub_query, "context": context})
        return response.content if hasattr(response, 'content') else str(response)


async def research_sub_query(
    sub_query: str, 
    agent: SubQueryAgent, 
    retriever: HybridRetriever
) -> Dict[str, Any]:
    """
    Self-RAG loop for a single sub-query.
    """
    print(f"[agent] Starting research for: '{sub_query}'")
    loop = asyncio.get_event_loop()
    
    # 1. Retrieve
    docs = await loop.run_in_executor(None, retriever.search, sub_query)
    
    # 2. Grade Relevance
    relevant_docs = await agent.grade_relevance(sub_query, docs)
    
    # 3. Fallback if irrelevant
    if not relevant_docs:
        print(f"[agent] All docs irrelevant for '{sub_query}'. Rewriting query...")
        rewritten_query = await agent.rewrite_query(sub_query)
        print(f"[agent] Rewritten query: '{rewritten_query}'")
        docs = await loop.run_in_executor(None, retriever.search, rewritten_query)
        relevant_docs = await agent.grade_relevance(rewritten_query, docs)
        
    # Format context
    parts = []
    for doc in relevant_docs:
        if doc.metadata.get("content_type") == "table":
            parts.append(f"[TABLE from {doc.metadata['source']}]\n{doc.page_content}")
        else:
            parts.append(doc.page_content)
    context_text = "\n\n---\n\n".join(parts)
    
    # 4. Generate Answer
    answer = await agent.answer(sub_query, context_text)
    
    # 5. Grade Hallucination
    is_grounded = await agent.check_hallucination(answer, context_text)
    if not is_grounded:
        print(f"[agent] Hallucination detected for '{sub_query}'. Regenerating...")
        # Simple fallback for hallucination: append a strict warning and regenerate once
        strict_context = context_text + "\n\nCRITICAL: Do not make up information. Use ONLY the above text."
        answer = await agent.answer(sub_query, strict_context)
    
    print(f"[agent] Completed research for: '{sub_query}'")
    return {
        "sub_query": sub_query,
        "answer": answer,
        "sources": [doc.metadata for doc in relevant_docs]
    }


async def run_parallel_research(
    sub_queries: List[str], 
    llm, 
    retriever: HybridRetriever
) -> List[Dict[str, Any]]:
    """Run research concurrently."""
    agent = SubQueryAgent(llm)
    tasks = [research_sub_query(query, agent, retriever) for query in sub_queries]
    return await asyncio.gather(*tasks)
