import marimo

__generated_with = "0.10.14"
app = marimo.App(width="medium")


@app.cell
def setup_imports():
    """Cell 0: Setup - Import all dependencies and configure environment."""
    import marimo as mo
    import os
    from dotenv import load_dotenv
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from typing import TypedDict, Annotated
    import operator
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import sys
    import io

    # Add examples directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

    from graph_helpers import create_linear_graph, create_react_graph, create_data_pipeline
    from visualization import (
        create_revenue_chart,
        create_product_comparison,
        create_region_chart,
        create_time_series,
        format_state_display,
        create_stats_table
    )

    # Load environment variables silently
    load_dotenv()

    # Configure LangSmith tracing (happens silently in background)
    # No output needed - just configuration
    return (
        mo, os, load_dotenv, StateGraph, END, HumanMessage, AIMessage,
        ChatOpenAI, tool, TypedDict, Annotated, operator, pd, px, go, io,
        create_linear_graph, create_react_graph, create_data_pipeline,
        create_revenue_chart, create_product_comparison, create_region_chart,
        create_time_series, format_state_display, create_stats_table
    )


@app.cell
def slide_1_introduction(mo):
    """Cell 1: Introduction - What is LangGraph and why it matters."""

    chain_graph = mo.mermaid("""
    graph LR
        A[Input] --> B[LLM]
        B --> C[Tool]
        C --> D[LLM]
        D --> E[Output]
        style A fill:#e1f5ff
        style E fill:#e1f5ff
    """)

    langgraph_diagram = mo.mermaid("""
    graph LR
        A[Input] --> B[Classify]
        B -->|Route A| C[Process]
        B -->|Route B| D[Analyze]
        C --> E{Decide}
        D --> E
        E -->|Loop| B
        E -->|End| F[Output]
        style A fill:#e1f5ff
        style F fill:#e1f5ff
        style E fill:#fff4e1
    """)

    mo.vstack([
        mo.md("# LangGraph: Building Stateful AI Agents"),
        mo.md("## From Linear Chains to Intelligent Graphs"),
        mo.md(""),
        mo.md("""
        **LangGraph** is a framework for building **stateful, multi-agent AI applications** using directed graphs.
        Unlike simple chains (sequential DAGs), graphs enable:

        - **Cycles**: Loops and iterative refinement
        - **Conditional Logic**: Dynamic routing based on state
        - **Complex Workflows**: Multi-step processes with branching
        - **State Management**: Persistent memory across steps
        """),
        mo.md("### Why Graphs > Chains?"),
        mo.md(""),
        mo.hstack([
            mo.vstack([
                mo.md("**Chain (DAG):**"),
                chain_graph,
                mo.md("_Linear execution only_")
            ]),
            mo.vstack([
                mo.md("**Graph (LangGraph):**"),
                langgraph_diagram,
                mo.md("_Cycles & conditional branching_")
            ])
        ], justify="space-around")
    ])


@app.cell
def slide_2_state_ui(mo):
    """Cell 2a: State Concepts - Create UI elements."""
    state_tabs = mo.ui.tabs({
        "TypedDict": "typed",
        "MessagesState": "messages"
    })
    return state_tabs,


@app.cell
def slide_2_state_content(mo, state_tabs):
    """Cell 2b: State Concepts - Display content based on UI selection."""

    typed_dict_ex = mo.md("""
    ### TypedDict State

    State as a **typed dictionary** - perfect for structured data workflows.

    ```python
    from typing import TypedDict

    class MyState(TypedDict):
        input: str
        step1_result: str
        step2_result: str
        final_output: str
    ```

    **Key Features:**
    - Type-safe with IDE support
    - Custom fields for your workflow
    - Clear data structure
    - Each node receives state, returns updates

    **Example Flow:**
    ```python
    def node1(state: MyState) -> dict:
        return {"step1_result": f"Processed: {state['input']}"}
    ```

    State acts like a **shared whiteboard** - nodes read from it and write updates to it.
    """)

    messages_ex = mo.md("""
    ### MessagesState

    Built-in state for **conversational agents** - handles message history automatically.

    ```python
    from langgraph.graph import MessagesState

    class MyState(MessagesState):
        user_name: str
        context: dict
    ```

    **Key Features:**
    - Pre-configured for chat applications
    - Automatic message list management
    - Works seamlessly with LangChain
    - Messages accumulate with `Annotated[list, operator.add]`

    Perfect for chatbots, assistants, and conversational AI!
    """)

    mo.vstack([
        mo.md("## Understanding State in LangGraph"),
        mo.md("_State is the shared memory of your graph - the 'whiteboard' where nodes read and write data._"),
        mo.md(""),
        state_tabs,
        mo.md(""),
        typed_dict_ex if state_tabs.value == "typed" else messages_ex
    ])


@app.cell
def slide_3_nodes_ui(mo):
    """Cell 3a: Nodes - Create UI elements."""
    node_input = mo.ui.text(value="Hello LangGraph", label="Input value:")

    # Editable node code
    node_code = mo.ui.text_area(
        value="""def process_node(state):
    input_val = state.get("input", "")
    # Modify this line to change behavior!
    processed = input_val.upper() + " - PROCESSED!"
    return {"output": processed}""",
        label="Edit the node code:",
        rows=6
    )

    return node_input, node_code


@app.cell
def slide_3_nodes_content(mo, node_input, node_code, TypedDict):
    """Cell 3b: Nodes - Display node processing."""

    class ExampleState(TypedDict):
        input: str
        output: str

    # Execute user's code
    try:
        exec_globals = {}
        exec(node_code.value, exec_globals)
        process_node = exec_globals['process_node']

        state_val = {"input": node_input.value}
        result_val = process_node(state_val)
        error_msg = None
    except Exception as e:
        result_val = {"output": f"Error: {str(e)}"}
        error_msg = str(e)

    mo.vstack([
        mo.md("## Nodes: The Workers of Your Graph"),
        mo.md("_Nodes are Python functions that receive state and return state updates._"),
        mo.md(""),
        mo.md("""
        ### What is a Node?

        A **node** is a function that:
        1. Receives the current state as input
        2. Performs some work (call LLM, process data, use tools)
        3. Returns a dictionary with state updates

        ### 🎨 Interactive Demo: Modify the Code!

        Try changing the node code below. For example:
        - Change `upper()` to `lower()`
        - Add `[::-1]` to reverse the string
        - Concatenate different text
        """),
        mo.md(""),
        node_code,
        mo.md(""),
        node_input,
        mo.md(""),
        mo.callout(f"**Error:** {error_msg}", kind="warn") if error_msg else mo.md(""),
        mo.md(f"""
        **Input State:**
        ```python
        {{"input": "{node_input.value}"}}
        ```

        **Node Processing:**
        ```python
        {node_code.value}
        ```

        **Output State:**
        ```python
        {{"output": "{result_val['output']}"}}
        ```
        """),
        mo.callout(
            "**Key Point**: Nodes don't modify state directly - they return updates that LangGraph merges into state.",
            kind="info"
        )
    ])


@app.cell
def slide_4_edges_ui(mo):
    """Cell 4a: Edges - Create UI elements."""
    edge_tabs = mo.ui.tabs({
        "Fixed Edges": "fixed",
        "Conditional Edges": "conditional"
    })
    return edge_tabs,


@app.cell
def slide_4_edges_content(mo, edge_tabs):
    """Cell 4b: Edges - Display content based on selection."""

    fixed_diagram = mo.mermaid("""
    graph LR
        A[Node A] --> B[Node B]
        B --> C[Node C]
        C --> D[END]
        style A fill:#90EE90
        style B fill:#87CEEB
        style C fill:#FFB6C1
        style D fill:#FFE4B5
    """)

    fixed_ex = mo.vstack([
        mo.md("### Fixed Edges"),
        mo.md("**Always go to the same next node** - simple sequential flow."),
        mo.md(""),
        fixed_diagram,
        mo.md(""),
        mo.md("""
        **Code:**
        ```python
        builder = StateGraph(MyState)
        builder.add_node("node_a", node_a_func)
        builder.add_node("node_b", node_b_func)
        builder.add_node("node_c", node_c_func)

        # Fixed edges - always go to next node
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", "node_c")
        builder.add_edge("node_c", END)
        ```

        **Use Cases:** Simple pipelines, sequential processing
        """)
    ])

    conditional_diagram = mo.mermaid("""
    graph LR
        A[Classify] --> B{Router}
        B -->|Math| C[Calculator]
        B -->|General| D[LLM]
        C --> E[END]
        D --> E
        style A fill:#90EE90
        style B fill:#FFE4B5
        style C fill:#87CEEB
        style D fill:#FFB6C1
        style E fill:#D3D3D3
    """)

    conditional_ex = mo.vstack([
        mo.md("### Conditional Edges"),
        mo.md("**Route based on state** - dynamic decision making!"),
        mo.md(""),
        conditional_diagram,
        mo.md(""),
        mo.md("""
        **Code:**
        ```python
        def router(state: MyState) -> str:
            \"\"\"Decide where to go next based on state.\"\"\"
            if state["question_type"] == "math":
                return "calculator"
            else:
                return "llm"

        builder = StateGraph(MyState)
        builder.add_node("classify", classify_func)
        builder.add_node("calculator", calc_func)
        builder.add_node("llm", llm_func)

        # Conditional edge - routes based on router function
        builder.add_conditional_edges(
            "classify",
            router,  # Function returns next node name
            {
                "calculator": "calculator",
                "llm": "llm"
            }
        )
        ```

        **Use Cases:** Intelligent routing, decision trees, adaptive workflows
        """)
    ])

    mo.vstack([
        mo.md("## Edges: Connecting the Flow"),
        mo.md("_Edges determine which node executes next - fixed or conditional based on state._"),
        mo.md(""),
        edge_tabs,
        mo.md(""),
        fixed_ex if edge_tabs.value == "fixed" else conditional_ex
    ])


@app.cell
def slide_5_graph_ui(mo):
    """Cell 5a: First Graph - Create UI elements."""
    graph_input = mo.ui.text(value="Hello LangGraph", label="Graph Input:")

    # Let users modify node behavior
    node1_transform = mo.ui.text(value="Step1", label="Node1 prefix:")
    node2_transform = mo.ui.text(value="Step2", label="Node2 prefix:")

    return graph_input, node1_transform, node2_transform


@app.cell
def slide_5_graph_content(mo, graph_input, node1_transform, node2_transform, StateGraph, END, TypedDict):
    """Cell 5b: First Graph - Execute and display."""

    class SimpleState(TypedDict):
        input: str
        step1: str
        step2: str
        output: str

    # Define nodes with user-configurable behavior
    def node1(state: SimpleState) -> dict:
        return {"step1": f"{node1_transform.value}: {state.get('input', '')}"}

    def node2(state: SimpleState) -> dict:
        return {"step2": f"{node2_transform.value}: {state.get('step1', '')}"}

    def node3(state: SimpleState) -> dict:
        return {"output": f"Done: {state.get('step2', '')}"}

    # Build the graph
    builder = StateGraph(SimpleState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_node("node3", node3)
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", "node3")
    builder.add_edge("node3", END)
    builder.set_entry_point("node1")

    linear_g = builder.compile()

    try:
        result_g = linear_g.invoke({"input": graph_input.value})
    except Exception as e:
        result_g = {"error": str(e)}

    graph_diagram = mo.mermaid("""
    graph LR
        START([START]) --> Node1[node1: Add Step1]
        Node1 --> Node2[node2: Add Step2]
        Node2 --> Node3[node3: Generate Output]
        Node3 --> END([END])
        style START fill:#e1f5ff
        style Node1 fill:#90EE90
        style Node2 fill:#87CEEB
        style Node3 fill:#FFB6C1
        style END fill:#D3D3D3
    """)

    mo.vstack([
        mo.md("## Your First LangGraph: Linear Workflow"),
        mo.md("_Build a simple 3-node sequential graph and see it in action!_"),
        mo.md(""),
        graph_diagram,
        mo.md(""),
        mo.md("""
        ### Node Definitions (Modify to change behavior!)

        ```python
        def node1(state):
            return {"step1": f"{node1_transform.value}: {state['input']}"}

        def node2(state):
            return {"step2": f"{node2_transform.value}: {state['step1']}"}

        def node3(state):
            return {"output": f"Done: {state['step2']}"}
        ```
        """),
        mo.md("### 🎨 Customize Node Behavior:"),
        mo.hstack([node1_transform, node2_transform]),
        mo.md(""),
        graph_input,
        mo.md(""),
        mo.md(f"""
        **Execution Result:**
        - Input: `{graph_input.value}`
        - Step 1: `{result_g.get('step1', 'N/A')}`
        - Step 2: `{result_g.get('step2', 'N/A')}`
        - Output: `{result_g.get('output', 'N/A')}`
        """),
        mo.callout(
            "**Try it!** Change the node prefixes above and see how the output changes!",
            kind="success"
        )
    ])


@app.cell
def slide_6_llm_ui(mo):
    """Cell 6a: LLM Integration - Create UI elements."""
    mock_mode_llm = mo.ui.checkbox(value=True, label="Mock mode (no API call)")
    question_llm = mo.ui.text_area(
        value="What is LangGraph and why is it useful?",
        label="Your question:"
    )
    return mock_mode_llm, question_llm


@app.cell
def slide_6_llm_content(mo, mock_mode_llm, question_llm, ChatOpenAI, HumanMessage, os):
    """Cell 6b: LLM Integration - Process and display."""

    def get_llm_response(query: str, use_mock: bool) -> str:
        if use_mock:
            return """LangGraph is a framework for building stateful, multi-agent applications using directed graphs.
It's useful because it enables complex workflows with cycles, conditional branching, and persistent state management."""
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("sk-your"):
                return "⚠️ Please add your OPENAI_API_KEY to .env file."
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0.7)
                response = llm.invoke([HumanMessage(content=query)])
                return response.content
            except Exception as e:
                return f"Error: {str(e)}"

    answer_llm = get_llm_response(question_llm.value, mock_mode_llm.value)

    mo.vstack([
        mo.md("## Adding Intelligence: LLM Integration"),
        mo.md("_Integrate language models into your graph nodes._"),
        mo.md(""),
        mo.md("""
        ### LLM Node Pattern

        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        def llm_node(state: MyState) -> dict:
            \"\"\"Node that calls an LLM.\"\"\"
            question = state["question"]

            # Call LLM
            llm = ChatOpenAI(model="gpt-4")
            response = llm.invoke([HumanMessage(content=question)])

            # Return state update
            return {"answer": response.content}
        ```
        """),
        mo.md("### Try It!"),
        mo.hstack([mock_mode_llm, mo.md(f"_{'Mock mode' if mock_mode_llm.value else '🌐 Real OpenAI API'}_")]),
        question_llm,
        mo.md(""),
        mo.md("**Answer:**"),
        mo.md(answer_llm)
    ])


@app.cell
def slide_7_routing_ui(mo):
    """Cell 7a: Conditional Routing - Create UI elements."""
    question_type_ui = mo.ui.dropdown(
        options=["math", "general"],
        value="math",
        label="Question type:"
    )
    user_question_ui = mo.ui.text(
        value="What is 15 * 23?",
        label="Question:"
    )
    use_mock_routing_ui = mo.ui.checkbox(value=True, label="Mock mode")
    return question_type_ui, user_question_ui, use_mock_routing_ui


@app.cell
def slide_7_routing_content(mo, question_type_ui, user_question_ui, use_mock_routing_ui, ChatOpenAI, HumanMessage, os):
    """Cell 7b: Conditional Routing - Process and display."""

    def handle_math_q(q: str) -> str:
        try:
            for op in ["*", "+", "-", "/"]:
                if op in q:
                    parts = q.split(op)
                    if len(parts) == 2:
                        num1 = int(''.join(filter(str.isdigit, parts[0])))
                        num2 = int(''.join(filter(str.isdigit, parts[1])))
                        if op == "*": return f"The answer is {num1 * num2}"
                        elif op == "+": return f"The answer is {num1 + num2}"
                        elif op == "-": return f"The answer is {num1 - num2}"
                        elif op == "/": return f"The answer is {num1 / num2}"
            return "Could not parse math expression."
        except:
            return "Error parsing."

    def handle_general_q(q: str, mock: bool) -> str:
        if mock:
            return "LangGraph enables building complex, stateful AI workflows!"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("sk-your"):
                return "⚠️ Add OPENAI_API_KEY to .env"
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0.7)
                response = llm.invoke([HumanMessage(content=q)])
                return response.content
            except Exception as e:
                return f"Error: {str(e)}"

    route_type = question_type_ui.value
    if route_type == "math":
        response_routing = handle_math_q(user_question_ui.value)
        active_path = "C"
    else:
        response_routing = handle_general_q(user_question_ui.value, use_mock_routing_ui.value)
        active_path = "D"

    routing_diagram = mo.mermaid(f"""
    graph LR
        A[Input Question] --> B[Classify]
        B -->|Math| C[Calculator]
        B -->|General| D[LLM]
        C --> E[Response]
        D --> E
        style {active_path} fill:#90EE90,stroke:#2d5016,stroke-width:3px
        style E fill:#FFE4B5
    """)

    mo.vstack([
        mo.md("## Conditional Routing: Intelligent Decision Making"),
        mo.md("_Route execution dynamically based on input classification._"),
        mo.md(""),
        routing_diagram,
        mo.md(""),
        mo.md("""
        ### Router Function

        ```python
        def router(state: MyState) -> str:
            \"\"\"Decide which node to execute next.\"\"\"
            question_type = classify(state["question"])

            if question_type == "math":
                return "calculator"  # Route to calculator
            else:
                return "llm"  # Route to LLM

        # Add conditional edge
        builder.add_conditional_edges("classify", router, {
            "calculator": "calculator",
            "llm": "llm"
        })
        ```
        """),
        mo.hstack([question_type_ui, use_mock_routing_ui]),
        user_question_ui,
        mo.md(""),
        mo.md(f"**Response:** {response_routing}"),
        mo.callout(
            f"Routed to: **{'Calculator' if route_type == 'math' else 'LLM'}** (highlighted in green)",
            kind="info"
        )
    ])


@app.cell
def slide_8_tools_ui(mo):
    """Cell 8a: Tools - Create UI elements."""
    tool_choice_ui = mo.ui.dropdown(
        options=["calculator", "string_reverser"],
        value="calculator",
        label="Select tool:"
    )
    tool_input_ui = mo.ui.text(value="15 * 23", label="Tool input:")
    return tool_choice_ui, tool_input_ui


@app.cell
def slide_8_tools_content(mo, tool_choice_ui, tool_input_ui, tool):
    """Cell 8b: Tools - Execute and display."""

    @tool
    def calc_tool(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            return f"Result: {eval(expression)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def reverse_tool(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    if tool_choice_ui.value == "calculator":
        tool_result = calc_tool.invoke({"expression": tool_input_ui.value})
    else:
        tool_result = reverse_tool.invoke({"text": tool_input_ui.value})

    mo.vstack([
        mo.md("## Tools: Extending Agent Capabilities"),
        mo.md("_Tools are functions that agents can invoke._"),
        mo.md(""),
        mo.md("""
        ### Defining a Tool

        ```python
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            \"\"\"Evaluate a mathematical expression.\"\"\"
            result = eval(expression)
            return f"Result: {result}"

        @tool
        def web_search(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            # Your search logic here
            return search_results
        ```

        ### Using Tools in Agents

        ```python
        # Bind tools to LLM
        tools = [calculator, web_search]
        llm = ChatOpenAI(model="gpt-4").bind_tools(tools)

        def agent_node(state):
            response = llm.invoke(state["messages"])
            # Agent decides whether to call tool
            if response.tool_calls:
                results = execute_tools(response.tool_calls)
                return {"messages": results}
            return {"messages": [response]}
        ```
        """),
        mo.md("### Try It!"),
        mo.hstack([tool_choice_ui, tool_input_ui]),
        mo.md(""),
        mo.md(f"**Tool Output:** `{tool_result}`"),
        mo.callout(
            "In real agents, the LLM decides *when* and *how* to use tools!",
            kind="success"
        )
    ])


@app.cell
def slide_9_react_ui(mo):
    """Cell 9a: ReAct Pattern - Create UI elements."""
    max_iterations_ui = mo.ui.slider(start=1, stop=5, value=3, label="Max iterations:")
    use_mock_react_ui = mo.ui.checkbox(value=True, label="Mock mode")
    return max_iterations_ui, use_mock_react_ui


@app.cell
def slide_9_react_content(mo, max_iterations_ui, use_mock_react_ui, create_react_graph):
    """Cell 9b: ReAct Pattern - Display."""

    react_g = create_react_graph(max_iter=max_iterations_ui.value, mock_mode=use_mock_react_ui.value)

    react_diagram = mo.mermaid("""
    graph LR
        A[Reason] --> B{Should Continue?}
        B -->|Yes| C[Use Tool]
        B -->|No| D[END]
        C --> E[Observe Result]
        E --> A
        style A fill:#FFE4B5
        style B fill:#D3D3D3
        style C fill:#90EE90
        style E fill:#87CEEB
        style D fill:#FFB6C1
    """)

    mo.vstack([
        mo.md("## Cycles & Loops: The ReAct Pattern"),
        mo.md("_Iterative reasoning with feedback loops - the foundation of agentic behavior._"),
        mo.md(""),
        react_diagram,
        mo.md(""),
        mo.md("""
        **ReAct** = **Reason** + **Act** + **Observe**

        ### How It Works

        ```python
        def reason_node(state):
            \"\"\"Agent thinks about what to do.\"\"\"
            llm = ChatOpenAI(model="gpt-4")
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state) -> str:
            \"\"\"Decide: continue, use tool, or finish.\"\"\"
            last_message = state["messages"][-1]

            if last_message.tool_calls:
                return "use_tool"
            elif state["iterations"] >= state["max_iterations"]:
                return "finish"
            else:
                return "reason"

        def tool_node(state):
            \"\"\"Execute tools and return results.\"\"\"
            results = execute_tools(last_message.tool_calls)
            return {"messages": results}

        # Build graph with cycle
        builder.add_conditional_edges("reason", should_continue, {
            "reason": "reason",      # Loop back!
            "use_tool": "use_tool",
            "finish": END
        })
        builder.add_edge("use_tool", "reason")  # Cycle!
        ```

        Cycles enable iterative refinement and multi-step reasoning.
        """),
        mo.md("### Configure ReAct"),
        max_iterations_ui,
        use_mock_react_ui,
        mo.md(""),
        mo.callout(
            f"Graph configured with max {max_iterations_ui.value} iterations. The agent will reason → act → observe → repeat!",
            kind="info"
        )
    ])


@app.cell
def slide_10_pipeline_architecture(mo):
    """Cell 10: Data Pipeline Architecture."""

    pipeline_diagram = mo.mermaid("""
    graph LR
        A[Load Data] --> B[Analyze]
        B --> C[Generate Insights]
        C --> D[Visualize]
        D --> E[END]
        style A fill:#FFE4B5
        style B fill:#87CEEB
        style C fill:#90EE90
        style D fill:#FFB6C1
        style E fill:#D3D3D3
    """)

    mo.vstack([
        mo.md("## Real-World Example: Data Analysis Pipeline"),
        mo.md("_A complete workflow that loads data, analyzes it, generates AI insights, and visualizes results._"),
        mo.md(""),
        pipeline_diagram,
        mo.md(""),
        mo.md("""
        ### Pipeline Stages

        #### 1. **Load Data** Node
        ```python
        def load_data(state):
            df = pd.read_csv("data.csv")
            return {"data": df}
        ```

        #### 2. **Analyze** Node
        ```python
        def analyze(state):
            df = state["data"]
            stats = {
                "total_revenue": df["revenue"].sum(),
                "top_product": df.groupby("product")["revenue"].sum().idxmax(),
                # ... more stats
            }
            return {"stats": stats}
        ```

        #### 3. **Generate Insights** Node
        ```python
        def generate_insights(state):
            stats = state["stats"]
            llm = ChatOpenAI(model="gpt-4")
            prompt = f"Analyze: {stats}"
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"insights": response.content}
        ```

        #### 4. **Visualize** Node
        ```python
        def visualize(state):
            df = state["data"]
            fig = px.bar(df.groupby("product")["revenue"].sum())
            return {"chart": fig}
        ```

        **Next slide**: See the full implementation in action!
        """)
    ])


@app.cell
def slide_11_pipeline_ui(mo):
    """Cell 11a: Data Pipeline - Create UI elements."""
    use_sample_data_ui = mo.ui.checkbox(value=True, label="Use sample data")
    use_mock_insights_ui = mo.ui.checkbox(value=True, label="Mock mode (no API)")
    return use_sample_data_ui, use_mock_insights_ui


@app.cell
def slide_11_pipeline_content(
    mo, use_sample_data_ui, use_mock_insights_ui, os,
    create_data_pipeline, create_revenue_chart,
    create_product_comparison, create_region_chart, create_stats_table
):
    """Cell 11b: Data Pipeline - Execute and display."""

    sample_path_val = os.path.join(os.path.dirname(__file__), "examples", "sample_data.csv")
    pipeline_val = create_data_pipeline(
        mock_mode=use_mock_insights_ui.value,
        sample_data_path=sample_path_val if use_sample_data_ui.value else None
    )

    try:
        pipeline_result = pipeline_val.invoke({})
        df_val = pipeline_result.get("data")
        stats_val = pipeline_result.get("stats", {})
        insights_val = pipeline_result.get("insights", "No insights")

        if df_val is not None and not df_val.empty:
            rev_chart = create_revenue_chart(df_val, chart_type="bar")
            prod_chart = create_product_comparison(df_val)
            reg_chart = create_region_chart(df_val)
            stats_tbl = create_stats_table(stats_val)
        else:
            rev_chart = None
            prod_chart = None
            reg_chart = None
            stats_tbl = "_No data_"
    except Exception as e:
        df_val = None
        insights_val = f"Error: {str(e)}"
        rev_chart = None
        prod_chart = None
        reg_chart = None
        stats_tbl = f"_Error: {str(e)}_"

    mo.vstack([
        mo.md("## Data Analysis Pipeline: Live Demo"),
        mo.md("_Full implementation with interactive controls._"),
        mo.md(""),
        mo.hstack([use_sample_data_ui, use_mock_insights_ui]),
        mo.md(""),
        mo.md(f"### 📊 Data: {len(df_val) if df_val is not None else 0} records"),
        mo.md("### 📈 Statistics"),
        mo.md(stats_tbl),
        mo.md(""),
        mo.md("### 🤖 AI Insights"),
        mo.md(insights_val),
        mo.md(""),
        mo.md("### 📉 Visualizations"),
        mo.ui.tabs({
            "Revenue": mo.ui.plotly(rev_chart) if rev_chart else mo.md("_No data_"),
            "Products": mo.ui.plotly(prod_chart) if prod_chart else mo.md("_No data_"),
            "Regions": mo.ui.plotly(reg_chart) if reg_chart else mo.md("_No data_"),
        })
    ])


@app.cell
def slide_12_langsmith(mo):
    """Cell 12: LangSmith Observability."""
    mo.md("""
    ## Observability: LangSmith Tracing

    _Debug, optimize, and understand your LangGraph applications._

    ### Configuration (.env)

    ```bash
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_your_key_here
    LANGCHAIN_PROJECT=langgraph-presentation
    ```

    ### What You Get

    - **Execution Traces**: See all nodes and LLM calls
    - **Token Usage**: Track costs per run
    - **Performance**: Identify bottlenecks
    - **Debugging**: Root cause analysis

    ### View Traces

    Visit [smith.langchain.com](https://smith.langchain.com) to see your traces!

    🎯 **Pro Tip**: Start tracing from day one!
    """)


@app.cell
def slide_13_advanced(mo):
    """Cell 13: Advanced Patterns."""
    mo.md("""
    ## Advanced Patterns & Next Steps

    ### 1. Subgraphs
    Nest graphs within nodes for modularity.

    ### 2. Multi-Agent Orchestration
    Multiple specialized agents collaborating.

    ### 3. Human-in-the-Loop
    Pause execution for human review.

    ### 4. Persistence & Checkpointing
    Save and restore graph state.

    ### 5. Error Handling & Retries
    Robust workflows with automatic retry.

    ### Resources

    - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
    - [LangChain Docs](https://python.langchain.com/)
    - [LangSmith](https://smith.langchain.com/)
    """)


@app.cell
def slide_14_summary(mo):
    """Cell 14: Summary & Next Steps."""
    mo.md("""
    # 🎯 Key Takeaways

    ## What We Learned

    1. **Graphs > Chains**: Cycles and conditionals enable complex workflows
    2. **Core Concepts**: State, Nodes, Edges, Graphs
    3. **Intelligent Agents**: LLMs, tools, and ReAct patterns
    4. **Production Patterns**: Data pipelines, observability, error handling

    ## 🚀 Next Steps

    **Run This Presentation:**
    ```bash
    uv run marimo run langgraph_deck.py
    ```

    **Build Your First Graph:**
    1. Start with a simple 3-node linear graph
    2. Add conditional routing
    3. Integrate an LLM
    4. Add tool use

    ## 📚 Resources

    - **LangGraph**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
    - **LangSmith**: [smith.langchain.com](https://smith.langchain.com/)
    - **Marimo**: [docs.marimo.io](https://docs.marimo.io/)

    **Happy building! 🚀**
    """)


if __name__ == "__main__":
    app.run()
