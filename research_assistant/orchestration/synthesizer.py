"""
Orchestration — Synthesizer
Synthesizes all sub-query findings into a cohesive, multi-source final report.
"""

from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate


class ReportSynthesizer:
    """Synthesizes multiple sub-answers into a final comprehensive report."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a senior academic researcher synthesizing a final report.
Based on the following research goal and the individual findings for each sub-query, 
write a comprehensive, well-structured final report.

Goal: {goal}

Requirements for the report:
1. Start with an Executive Summary.
2. Synthesize the findings across sub-queries (do not just list them one by one).
3. Identify patterns, contradictions, or gaps in the literature.
4. Conclude with actionable insights or next steps.
5. You MUST include inline citations to the sources mentioned in the findings (e.g., [Source: Name]).
6. Format in clean Markdown with appropriate headers.

Findings:
{findings}

Final Report:""",
            input_variables=["goal", "findings"]
        )

    async def synthesize(self, goal: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate the final report from sub-query results.
        """
        print(f"[synthesizer] Synthesizing final report for: '{goal}'")
        
        # Format the findings into a single block
        findings_parts = []
        for res in results:
            q = res["sub_query"]
            a = res["answer"]
            findings_parts.append(f"### Finding for Sub-Query: {q}\n{a}\n")
            
        findings_text = "\n\n".join(findings_parts)
        
        chain = self.prompt | self.llm
        
        response = await chain.ainvoke({
            "goal": goal,
            "findings": findings_text
        })
        
        print("[synthesizer] Report synthesis complete.")
        return response.content if hasattr(response, 'content') else str(response)
