"""
Visualization — Chart Generator
Extracts numerical relationships from text and generates Python-based visualisations (matplotlib).
"""

import re
import io
import base64
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ChartData(BaseModel):
    """Structured output for chart extraction."""
    has_data: bool = Field(description="True if the text contains comparable numerical data suitable for a bar chart.")
    title: str = Field(description="Title of the chart.")
    xlabel: str = Field(description="Label for the X axis.")
    ylabel: str = Field(description="Label for the Y axis.")
    labels: list[str] = Field(description="List of labels for the X axis (e.g., model names, years).")
    values: list[float] = Field(description="List of numerical values corresponding to the labels.")


class ChartGenerator:
    """Extracts data and generates a matplotlib chart if applicable."""

    def __init__(self, llm):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(ChartData)
        
        self.prompt = PromptTemplate(
            template="""Analyze the following research report and determine if there is numerical data that can be visualized as a bar chart.
Look for comparisons, benchmarks, percentages, or performance metrics across different entities (like models, methods, or years).

If suitable data exists, extract it. The 'labels' and 'values' arrays MUST be the exactly the same length.
If no suitable numerical comparison data exists, set 'has_data' to false.

Report Text:
{text}
""",
            input_variables=["text"]
        )

    async def extract_and_plot(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Attempts to extract data and generate a plot.
        Returns: (success, base64_image_data, explanation_message)
        """
        print("[chart_generator] Analyzing text for visualization data...")
        
        chain = self.prompt | self.structured_llm
        
        try:
            data: ChartData = await chain.ainvoke({"text": text})
        except Exception as e:
            print(f"[chart_generator] Failed to extract data: {e}")
            return False, None, "Failed to parse numerical data from the report."

        if not data.has_data or not data.labels or not data.values:
            print("[chart_generator] No suitable visualization data found.")
            return False, None, "No comparative numerical data found to visualize."
            
        if len(data.labels) != len(data.values):
            print("[chart_generator] Data length mismatch.")
            return False, None, "Extracted labels and values did not match in length."

        print(f"[chart_generator] Generating plot: '{data.title}'")
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.bar(data.labels, data.values, color='skyblue', edgecolor='black')
        plt.title(data.title, fontsize=14)
        plt.xlabel(data.xlabel, fontsize=12)
        plt.ylabel(data.ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return True, img_base64, "Generated visualization from extracted data."
