"""Visualization utilities for the LangGraph presentation."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from typing import Any


def create_revenue_chart(df: pd.DataFrame, chart_type: str = "bar") -> go.Figure:
    """
    Create a revenue chart from DataFrame.

    Args:
        df: DataFrame with 'product' and 'revenue' columns
        chart_type: 'bar' or 'line'

    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Aggregate revenue by product
    product_revenue = df.groupby("product")["revenue"].sum().reset_index()
    product_revenue = product_revenue.sort_values("revenue", ascending=False)

    if chart_type == "bar":
        fig = px.bar(
            product_revenue,
            x="product",
            y="revenue",
            title="Revenue by Product",
            labels={"revenue": "Total Revenue ($)", "product": "Product"},
            color="revenue",
            color_continuous_scale="Blues"
        )
    else:
        fig = px.line(
            product_revenue,
            x="product",
            y="revenue",
            title="Revenue by Product",
            labels={"revenue": "Total Revenue ($)", "product": "Product"},
            markers=True
        )

    fig.update_layout(
        xaxis_title="Product",
        yaxis_title="Revenue ($)",
        hovermode="x unified"
    )

    return fig


def create_product_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive product comparison chart.

    Args:
        df: DataFrame with product sales data

    Returns:
        Plotly Figure object with multiple subplots
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Aggregate data
    product_stats = df.groupby("product").agg({
        "revenue": "sum",
        "quantity": "sum"
    }).reset_index()

    product_stats = product_stats.sort_values("revenue", ascending=True)

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=product_stats["product"],
        x=product_stats["revenue"],
        name="Revenue",
        orientation="h",
        marker=dict(color="#1f77b4")
    ))

    fig.update_layout(
        title="Product Performance",
        xaxis_title="Total Revenue ($)",
        yaxis_title="Product",
        height=400,
        showlegend=False,
        hovermode="y unified"
    )

    return fig


def create_region_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a regional performance chart.

    Args:
        df: DataFrame with regional sales data

    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Aggregate by region
    region_revenue = df.groupby("region")["revenue"].sum().reset_index()

    fig = px.pie(
        region_revenue,
        values="revenue",
        names="region",
        title="Revenue Distribution by Region",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    return fig


def create_time_series(df: pd.DataFrame) -> go.Figure:
    """
    Create a time series revenue chart.

    Args:
        df: DataFrame with date and revenue columns

    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date
    daily_revenue = df.groupby("date")["revenue"].sum().reset_index()
    daily_revenue = daily_revenue.sort_values("date")

    fig = px.line(
        daily_revenue,
        x="date",
        y="revenue",
        title="Revenue Over Time",
        labels={"revenue": "Daily Revenue ($)", "date": "Date"},
        markers=True
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified"
    )

    return fig


def format_state_display(state: dict, max_length: int = 500) -> str:
    """
    Format state dictionary for display in markdown.

    Args:
        state: State dictionary to format
        max_length: Maximum length for string values

    Returns:
        Formatted markdown string
    """
    def truncate(value: Any, max_len: int = max_length) -> str:
        """Truncate long values."""
        str_val = str(value)
        if len(str_val) > max_len:
            return str_val[:max_len] + "..."
        return str_val

    def format_value(value: Any) -> str:
        """Format a single value."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return f"DataFrame({value.shape[0]} rows × {value.shape[1] if hasattr(value, 'shape') and len(value.shape) > 1 else 1} cols)"
        elif isinstance(value, dict):
            return json.dumps(value, indent=2, default=str)
        elif isinstance(value, list):
            if len(value) > 5:
                return f"[{', '.join(map(str, value[:5]))}, ... ({len(value)} items)]"
            return str(value)
        else:
            return truncate(value)

    lines = ["```python"]
    lines.append("{")

    for key, value in state.items():
        formatted_value = format_value(value)
        lines.append(f"  '{key}': {formatted_value},")

    lines.append("}")
    lines.append("```")

    return "\n".join(lines)


def create_graph_diagram(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    """
    Create a mermaid diagram of the graph structure.

    Args:
        nodes: List of node names
        edges: List of (source, target) tuples

    Returns:
        Mermaid diagram string
    """
    lines = ["```mermaid", "graph LR"]

    # Add nodes
    for node in nodes:
        lines.append(f"    {node}[{node}]")

    # Add edges
    for source, target in edges:
        lines.append(f"    {source} --> {target}")

    lines.append("```")

    return "\n".join(lines)


def create_stats_table(stats: dict) -> str:
    """
    Create a formatted markdown table from statistics dictionary.

    Args:
        stats: Dictionary of statistics

    Returns:
        Markdown table string
    """
    if not stats:
        return "_No statistics available_"

    lines = ["| Metric | Value |", "|--------|-------|"]

    for key, value in stats.items():
        # Format the key (convert snake_case to Title Case)
        formatted_key = key.replace("_", " ").title()

        # Format the value
        if isinstance(value, float):
            if "revenue" in key.lower() or "price" in key.lower():
                formatted_value = f"${value:,.2f}"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        elif isinstance(value, tuple):
            formatted_value = f"{value[0]} to {value[1]}"
        else:
            formatted_value = str(value)

        lines.append(f"| {formatted_key} | {formatted_value} |")

    return "\n".join(lines)
