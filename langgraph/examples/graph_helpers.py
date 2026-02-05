"""LangGraph construction utilities for the presentation."""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import pandas as pd
import os


# State definitions for different graph examples


class SimpleState(TypedDict):
    """State for simple linear graph example."""
    input: str
    step1: str
    step2: str
    output: str


class ReActState(TypedDict):
    """State for ReAct pattern graph."""
    messages: Annotated[list, operator.add]
    iterations: int
    max_iterations: int
    final_answer: str


class DataPipelineState(TypedDict):
    """State for data analysis pipeline."""
    data: pd.DataFrame | None
    stats: dict | None
    insights: str
    chart_data: dict | None


# Helper functions for graph construction


def create_linear_graph():
    """
    Create a simple 3-node linear graph for demonstrating basic flow.

    Graph: node1 -> node2 -> node3 -> END
    """

    def node1(state: SimpleState) -> dict:
        """First node: process input."""
        return {"step1": f"Step1: {state.get('input', '')}"}

    def node2(state: SimpleState) -> dict:
        """Second node: process step1."""
        return {"step2": f"Step2: {state.get('step1', '')}"}

    def node3(state: SimpleState) -> dict:
        """Third node: generate final output."""
        return {"output": f"Done: {state.get('step2', '')}"}

    # Build the graph
    builder = StateGraph(SimpleState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_node("node3", node3)

    # Add edges
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", "node3")
    builder.add_edge("node3", END)

    # Set entry point
    builder.set_entry_point("node1")

    return builder.compile()


def create_react_graph(max_iter: int = 3, mock_mode: bool = True):
    """
    Create a ReAct pattern graph with reasoning and tool use.

    Graph: reason -> [use_tool or finish] -> observe -> reason (cycle)

    Args:
        max_iter: Maximum iterations before stopping
        mock_mode: If True, use mock LLM responses
    """

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Warning: eval is unsafe in production - for demo only
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def reason_node(state: ReActState) -> dict:
        """Agent reasons about what to do next."""
        messages = state.get("messages", [])
        iterations = state.get("iterations", 0)

        if mock_mode:
            # Mock reasoning
            if iterations == 0:
                thought = "I need to calculate 15 * 23"
                return {
                    "messages": [AIMessage(content=f"Thought: {thought}")],
                    "iterations": iterations + 1
                }
            else:
                return {
                    "messages": [AIMessage(content="I have the answer: 345")],
                    "final_answer": "345",
                    "iterations": iterations + 1
                }
        else:
            # Real LLM reasoning
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            response = llm.invoke(messages + [HumanMessage(content="What should I do next?")])
            return {
                "messages": [response],
                "iterations": iterations + 1
            }

    def should_continue(state: ReActState) -> str:
        """Decide whether to use tool, finish, or continue reasoning."""
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 3)
        final_answer = state.get("final_answer", "")

        if final_answer or iterations >= max_iterations:
            return "finish"
        elif iterations > 0:
            return "use_tool"
        else:
            return "reason"

    def tool_node(state: ReActState) -> dict:
        """Execute the calculator tool."""
        result = calculator("15 * 23")
        return {
            "messages": [AIMessage(content=f"Tool result: {result}")]
        }

    # Build the graph
    builder = StateGraph(ReActState)
    builder.add_node("reason", reason_node)
    builder.add_node("use_tool", tool_node)

    # Add conditional edges
    builder.add_conditional_edges(
        "reason",
        should_continue,
        {
            "reason": "reason",
            "use_tool": "use_tool",
            "finish": END
        }
    )
    builder.add_edge("use_tool", "reason")

    # Set entry point
    builder.set_entry_point("reason")

    return builder.compile()


def create_data_pipeline(mock_mode: bool = True, sample_data_path: str = None):
    """
    Create a 4-node data analysis pipeline.

    Graph: load_data -> analyze -> generate_insights -> visualize -> END

    Args:
        mock_mode: If True, use mock LLM responses
        sample_data_path: Path to CSV file (if None, creates sample data)
    """

    def load_data(state: DataPipelineState) -> dict:
        """Load data from CSV file."""
        if sample_data_path and os.path.exists(sample_data_path):
            df = pd.read_csv(sample_data_path)
        else:
            # Create sample data if file doesn't exist
            import numpy as np
            from datetime import datetime, timedelta

            dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
            products = ['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y']
            regions = ['North', 'South', 'East', 'West']

            data = []
            np.random.seed(42)
            for date in dates:
                for _ in range(3):
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'product': np.random.choice(products),
                        'region': np.random.choice(regions),
                        'revenue': np.random.uniform(100, 1000),
                        'quantity': np.random.randint(1, 20)
                    })

            df = pd.DataFrame(data)

        return {"data": df}

    def analyze(state: DataPipelineState) -> dict:
        """Calculate descriptive statistics."""
        df = state.get("data")
        if df is None or df.empty:
            return {"stats": {}}

        stats = {
            "total_revenue": float(df["revenue"].sum()),
            "avg_revenue": float(df["revenue"].mean()),
            "median_revenue": float(df["revenue"].median()),
            "total_quantity": int(df["quantity"].sum()),
            "num_transactions": len(df),
            "top_product": df.groupby("product")["revenue"].sum().idxmax(),
            "top_product_revenue": float(df.groupby("product")["revenue"].sum().max()),
            "top_region": df.groupby("region")["revenue"].sum().idxmax(),
            "date_range": (str(df["date"].min()), str(df["date"].max()))
        }

        return {"stats": stats}

    def generate_insights(state: DataPipelineState) -> dict:
        """Use LLM to generate insights from statistics."""
        stats = state.get("stats", {})

        if not stats:
            return {"insights": "No data to analyze."}

        if mock_mode:
            # Mock insights
            insights = f"""
## Key Insights from Data Analysis

**Revenue Performance:**
- Total revenue generated: ${stats['total_revenue']:,.2f}
- Average transaction value: ${stats['avg_revenue']:.2f}
- Median transaction value: ${stats['median_revenue']:.2f}

**Top Performers:**
- Best product: **{stats['top_product']}** with ${stats['top_product_revenue']:,.2f} in revenue
- Best region: **{stats['top_region']}**

**Activity:**
- Total transactions: {stats['num_transactions']}
- Total units sold: {stats['total_quantity']}
- Data period: {stats['date_range'][0]} to {stats['date_range'][1]}

**Recommendations:**
1. Focus marketing efforts on {stats['top_product']} as it's the top revenue generator
2. Investigate opportunities to replicate {stats['top_region']}'s success in other regions
3. Consider bundling strategies to increase average transaction value
"""
        else:
            # Real LLM insights
            import json
            llm = ChatOpenAI(model="gpt-4", temperature=0.7)
            prompt = f"""
You are a data analyst. Analyze these sales statistics and provide 3-5 key insights
with actionable recommendations. Be specific and data-driven.

Statistics:
{json.dumps(stats, indent=2, default=str)}

Format your response in markdown with clear sections.
"""
            response = llm.invoke([HumanMessage(content=prompt)])
            insights = response.content

        return {"insights": insights}

    def visualize(state: DataPipelineState) -> dict:
        """Prepare data for visualization."""
        df = state.get("data")
        if df is None or df.empty:
            return {"chart_data": {}}

        # Aggregate data for charts
        product_revenue = df.groupby("product")["revenue"].sum().reset_index()
        product_revenue = product_revenue.sort_values("revenue", ascending=False)

        region_revenue = df.groupby("region")["revenue"].sum().reset_index()

        chart_data = {
            "product_revenue": product_revenue.to_dict(orient="records"),
            "region_revenue": region_revenue.to_dict(orient="records"),
        }

        return {"chart_data": chart_data}

    # Build the graph
    builder = StateGraph(DataPipelineState)
    builder.add_node("load_data", load_data)
    builder.add_node("analyze", analyze)
    builder.add_node("generate_insights", generate_insights)
    builder.add_node("visualize", visualize)

    # Add edges
    builder.add_edge("load_data", "analyze")
    builder.add_edge("analyze", "generate_insights")
    builder.add_edge("generate_insights", "visualize")
    builder.add_edge("visualize", END)

    # Set entry point
    builder.set_entry_point("load_data")

    return builder.compile()
