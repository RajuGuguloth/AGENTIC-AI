"""
Orchestration — Planner
Uses LangChain to decompose a complex research goal into targeted sub-queries.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from config import Config


class SubQueriesOutput(BaseModel):
    """Structured output for the LLM planner."""
    sub_queries: List[str] = Field(
        description="List of specific, targeted research sub-queries to investigate the main goal."
    )


class ResearchPlanner:
    """Decomposes a broad research goal into sub-queries using an LLM."""

    def __init__(self, llm):
        """
        Args:
            llm: A LangChain ChatModel instance.
        """
        self.llm = llm
        
        # We use the structured output capability of modern LLMs
        self.structured_llm = self.llm.with_structured_output(SubQueriesOutput)
        
        self.prompt = PromptTemplate(
            template="""You are an expert academic research assistant.
Your task is to break down the following complex research goal into highly targeted sub-queries.
These sub-queries will be executed independently against a retrieval engine (Hybrid FAISS + BM25).

Research Goal: {goal}

Generate up to {max_queries} sub-queries. Each sub-query should:
1. Focus on a specific aspect of the main goal.
2. Be self-contained and highly searchable.
3. Not overlap significantly with other sub-queries.

Return the sub-queries as a list.
""",
            input_variables=["goal", "max_queries"]
        )

    async def plan(self, goal: str, max_queries: int = None) -> List[str]:
        """
        Generate sub-queries for the given goal.
        
        Args:
            goal: The main research question/goal.
            max_queries: Max number of sub-queries to generate.
            
        Returns:
            List of sub-query strings.
        """
        max_queries = max_queries or Config.MAX_SUB_QUERIES
        print(f"[planner] Decomposing goal: '{goal}' into max {max_queries} sub-queries...")
        
        chain = self.prompt | self.structured_llm
        
        result: SubQueriesOutput = await chain.ainvoke({
            "goal": goal,
            "max_queries": max_queries
        })
        
        queries = result.sub_queries[:max_queries]
        print(f"[planner] Generated {len(queries)} sub-queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
            
        return queries
