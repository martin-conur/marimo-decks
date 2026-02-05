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
    _output = mo.md("""
    # LangGraph: Building Stateful AI Agents

    ## From Linear Chains to Intelligent Graphs

    ### What is LangGraph?

    **LangGraph** is a framework for building **stateful, multi-agent AI applications** using directed graphs.
    Unlike simple chains (sequential DAGs), graphs enable:

    - **Cycles**: Loops and iterative refinement
    - **Conditional Logic**: Dynamic routing based on state
    - **Complex Workflows**: Multi-step processes with branching
    - **State Management**: Persistent memory across steps

    ### Why Graphs > Chains?

    | **Chains (DAGs)** | **Graphs (LangGraph)** |
    |-------------------|------------------------|
    | Linear execution only | Cycles and loops supported |
    | No conditional branching | Dynamic routing |
    | Limited state management | Rich, typed state |
    | Simple workflows | Complex, adaptive workflows |

    ### Visual Comparison

    **Chain (DAG):**
    ```
    Input → LLM → Tool → LLM → Output
    ```

    **Graph (LangGraph):**
    ```
    Input → Classify → [Route A] → Process → Decide → [Loop back or End]
                     ↓                              ↑
                   [Route B] → Analyze ────────────┘
    ```

    ---

    **Let's dive into the core concepts!**
    """)
    return _output,


@app.cell
def slide_2_state(mo):
    """Cell 2: State Concepts - Understanding shared memory in graphs."""

    # Interactive tabs to switch between examples
    _state_tabs = mo.ui.tabs({
        "TypedDict": "typed",
        "MessagesState": "messages"
    })

    _typed_dict_example = mo.md("""
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
    # Node 1 receives state, returns update
    def node1(state: MyState) -> dict:
        return {"step1_result": f"Processed: {state['input']}"}

    # Node 2 builds on previous results
    def node2(state: MyState) -> dict:
        return {"step2_result": f"Enhanced: {state['step1_result']}"}
    ```

    State acts like a **shared whiteboard** - nodes read from it and write updates to it.
    """)

    _messages_example = mo.md("""
    ### MessagesState

    Built-in state for **conversational agents** - handles message history automatically.

    ```python
    from langgraph.graph import MessagesState

    # MessagesState already has a 'messages' field
    class MyState(MessagesState):
        # You can add additional fields
        user_name: str
        context: dict
    ```

    **Key Features:**
    - Pre-configured for chat applications
    - Automatic message list management
    - Works seamlessly with LangChain
    - Messages accumulate with `Annotated[list, operator.add]`

    **Example:**
    ```python
    def agent_node(state: MyState) -> dict:
        messages = state["messages"]
        llm = ChatOpenAI(model="gpt-4")
        response = llm.invoke(messages)
        return {"messages": [response]}  # Appends to message list
    ```

    Perfect for chatbots, assistants, and conversational AI!
    """)

    _content = mo.vstack([
        mo.md("## Understanding State in LangGraph"),
        mo.md("_State is the shared memory of your graph - the 'whiteboard' where nodes read and write data._"),
        mo.md(""),
        _state_tabs,
        mo.md(""),
        _typed_dict_example if _state_tabs.value == "typed" else _messages_example
    ])

    return _state_tabs, _content


@app.cell
def slide_3_nodes(mo, TypedDict):
    """Cell 3: Nodes - Functions that transform state."""

    # Interactive input to demonstrate node processing
    node_input = mo.ui.text(value="Hello LangGraph", label="Input value:")

    # Define example state and node
    class ExampleState(TypedDict):
        input: str
        output: str

    def _example_node(state: ExampleState) -> dict:
        """Node that processes the input."""
        input_val = state.get("input", "")
        processed = input_val.upper() + " - Processed by node!"
        return {"output": processed}

    # Execute the node
    _state = {"input": node_input.value}
    _result = _example_node(_state)

    _content = mo.vstack([
        mo.md("## Nodes: The Workers of Your Graph"),
        mo.md("_Nodes are Python functions that receive state and return state updates._"),
        mo.md(""),
        mo.md("""
        ### What is a Node?

        A **node** is a function that:
        1. Receives the current state as input
        2. Performs some work (call LLM, process data, use tools)
        3. Returns a dictionary with state updates

        ### Node Structure

        ```python
        def my_node(state: MyState) -> dict:
            \"\"\"A simple node function.\"\"\"
            # Read from state
            input_data = state["input"]

            # Do some work
            result = process(input_data)

            # Return state update
            return {"output": result}
        ```

        ### Try It Live!

        Change the input below and see how the node processes it:
        """),
        mo.md(""),
        node_input,
        mo.md(""),
        mo.md(f"""
        **Input State:**
        ```python
        {{"input": "{node_input.value}"}}
        ```

        **Node Processing:**
        ```python
        def example_node(state):
            input_val = state["input"]
            processed = input_val.upper() + " - Processed by node!"
            return {{"output": processed}}
        ```

        **Output State:**
        ```python
        {{"output": "{_result['output']}"}}
        ```
        """),
        mo.md(""),
        mo.callout(
            "**Key Point**: Nodes don't modify state directly - they return updates that LangGraph merges into state.",
            kind="info"
        )
    ])

    return node_input, _content


@app.cell
def slide_4_edges(mo):
    """Cell 4: Edges - Control flow and routing."""

    # Tabs to show different edge types
    _edge_tabs = mo.ui.tabs({
        "Fixed Edges": "fixed",
        "Conditional Edges": "conditional"
    })

    _fixed_example = mo.md("""
    ### Fixed Edges

    **Always go to the same next node** - simple sequential flow.

    ```mermaid
    graph LR
        A[Node A] --> B[Node B]
        B --> C[Node C]
        C --> D[END]
    ```

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

    **Use Cases:**
    - Simple pipelines
    - Sequential processing
    - Predictable workflows
    """)

    _conditional_example = mo.md("""
    ### Conditional Edges

    **Route based on state** - dynamic decision making!

    ```mermaid
    graph LR
        A[Classify] --> B{Router}
        B -->|Math| C[Calculator]
        B -->|General| D[LLM]
        C --> E[END]
        D --> E
    ```

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
        router,  # Function that returns next node name
        {
            "calculator": "calculator",
            "llm": "llm"
        }
    )
    ```

    **Use Cases:**
    - Intelligent routing
    - Decision trees
    - Adaptive workflows
    - Error handling paths
    """)

    _content = mo.vstack([
        mo.md("## Edges: Connecting the Flow"),
        mo.md("_Edges determine which node executes next - fixed or conditional based on state._"),
        mo.md(""),
        _edge_tabs,
        mo.md(""),
        _fixed_example if _edge_tabs.value == "fixed" else _conditional_example,
        mo.md(""),
        mo.callout(
            "**Pro Tip**: Use fixed edges for simple flows, conditional edges for intelligent routing!",
            kind="success"
        )
    ])

    return _edge_tabs, _content


@app.cell
def slide_5_first_graph(mo, create_linear_graph):
    """Cell 5: First Graph - Build a simple linear workflow."""

    # Interactive input
    graph_input = mo.ui.text(value="Hello LangGraph", label="Graph Input:")

    # Create the graph
    linear_graph = create_linear_graph()

    # Execute the graph
    try:
        _result = linear_graph.invoke({"input": graph_input.value})
    except Exception as e:
        _result = {"error": str(e)}

    # Create visual diagram
    _diagram = mo.md("""
    ```mermaid
    graph LR
        START([START]) --> Node1[node1: Add Step1]
        Node1 --> Node2[node2: Add Step2]
        Node2 --> Node3[node3: Generate Output]
        Node3 --> END([END])
    ```
    """)

    _content = mo.vstack([
        mo.md("## Your First LangGraph: Linear Workflow"),
        mo.md("_Let's build a simple 3-node sequential graph and see it in action!_"),
        mo.md(""),
        _diagram,
        mo.md(""),
        mo.md("### Graph Structure"),
        mo.md("""
        ```python
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class SimpleState(TypedDict):
            input: str
            step1: str
            step2: str
            output: str

        def node1(state): return {"step1": f"Step1: {state['input']}"}
        def node2(state): return {"step2": f"Step2: {state['step1']}"}
        def node3(state): return {"output": f"Done: {state['step2']}"}

        builder = StateGraph(SimpleState)
        builder.add_node("node1", node1)
        builder.add_node("node2", node2)
        builder.add_node("node3", node3)
        builder.add_edge("node1", "node2")
        builder.add_edge("node2", "node3")
        builder.add_edge("node3", END)
        builder.set_entry_point("node1")

        graph = builder.compile()
        ```
        """),
        mo.md("### Try It Live!"),
        graph_input,
        mo.md(""),
        mo.md(f"""
        **Execution Result:**
        ```python
        Input: "{graph_input.value}"
        Step 1: "{_result.get('step1', 'N/A')}"
        Step 2: "{_result.get('step2', 'N/A')}"
        Output: "{_result.get('output', 'N/A')}"
        ```
        """),
        mo.callout(
            "**What happened?** Each node processed the state sequentially, adding its contribution!",
            kind="success"
        )
    ])

    return graph_input, linear_graph, _content


@app.cell
def slide_6_llm_integration(mo, ChatOpenAI, HumanMessage, os):
    """Cell 6: LLM Integration - Adding AI to your nodes."""

    # Interactive controls
    mock_mode = mo.ui.checkbox(value=True, label="Mock mode (no API call)")
    question_llm = mo.ui.text_area(
        value="What is LangGraph and why is it useful?",
        label="Your question:"
    )

    # LLM node function
    def _llm_node(query: str, use_mock: bool) -> str:
        """Node that calls an LLM."""
        if use_mock:
            return """LangGraph is a framework for building stateful, multi-agent applications using directed graphs.
It's useful because it enables complex workflows with cycles, conditional branching, and persistent state management -
going beyond simple linear chains to create truly adaptive AI systems."""
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("sk-your"):
                return "⚠️ Please add your OPENAI_API_KEY to .env file to use real LLM mode."
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0.7)
                response = llm.invoke([HumanMessage(content=query)])
                return response.content
            except Exception as e:
                return f"Error calling LLM: {str(e)}"

    # Execute
    _answer = _llm_node(question_llm.value, mock_mode.value)

    _content = mo.vstack([
        mo.md("## Adding Intelligence: LLM Integration"),
        mo.md("_Integrate language models into your graph nodes for AI-powered workflows._"),
        mo.md(""),
        mo.md("""
        ### LLM Node Pattern

        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        def llm_node(state: MyState) -> dict:
            \"\"\"Node that calls an LLM.\"\"\"
            # Get input from state
            question = state["question"]

            # Call LLM
            llm = ChatOpenAI(model="gpt-4")
            response = llm.invoke([HumanMessage(content=question)])

            # Return state update
            return {"answer": response.content}
        ```

        ### Interactive Demo

        Try asking a question! Toggle mock mode on/off to use real or simulated LLM responses.
        """),
        mo.md(""),
        mo.hstack([mock_mode, mo.md(f"_{'Using mock response' if mock_mode.value else 'Using real OpenAI API'}_")]),
        mo.md(""),
        question_llm,
        mo.md(""),
        mo.md("**Answer:**"),
        mo.md(_answer),
        mo.md(""),
        mo.callout(
            f"{'💡 Mock mode is active - no API calls made!' if mock_mode.value else '🌐 Real LLM mode - using OpenAI API'}",
            kind="info" if mock_mode.value else "warn"
        )
    ])

    return mock_mode, question_llm, _content


@app.cell
def slide_7_conditional_routing(mo, ChatOpenAI, HumanMessage, os):
    """Cell 7: Conditional Routing - Dynamic decision making."""

    # Interactive controls
    question_type = mo.ui.dropdown(
        options=["math", "general"],
        value="math",
        label="Question type:"
    )
    user_question = mo.ui.text(
        value="What is 15 * 23?",
        label="Question:"
    )
    use_mock_routing = mo.ui.checkbox(value=True, label="Mock mode")

    # Router function
    def _classify_question(question: str) -> str:
        """Classify question as math or general."""
        keywords = ["calculate", "+", "-", "*", "/", "×", "÷", "multiply", "divide", "sum", "what is"]
        has_numbers = any(char.isdigit() for char in question)
        has_math_keyword = any(kw in question.lower() for kw in keywords)

        if has_numbers and has_math_keyword:
            return "math"
        return "general"

    # Process based on type
    def _handle_math(question: str) -> str:
        """Handle math questions."""
        # Simple eval for demo - unsafe in production!
        try:
            # Extract expression
            for op in ["*", "+", "-", "/"]:
                if op in question:
                    parts = question.split(op)
                    if len(parts) == 2:
                        num1 = int(''.join(filter(str.isdigit, parts[0])))
                        num2 = int(''.join(filter(str.isdigit, parts[1])))
                        if op == "*": result = num1 * num2
                        elif op == "+": result = num1 + num2
                        elif op == "-": result = num1 - num2
                        elif op == "/": result = num1 / num2
                        return f"The answer is {result}"
            return "I can handle math questions like: What is 5 + 3?"
        except:
            return "Could not parse math expression."

    def _handle_general(question: str, mock: bool) -> str:
        """Handle general questions."""
        if mock:
            return "LangGraph enables building complex, stateful AI workflows with cycles and conditional branching - perfect for adaptive systems!"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("sk-your"):
                return "⚠️ Add OPENAI_API_KEY to .env for real LLM mode."
            try:
                llm = ChatOpenAI(model="gpt-4", temperature=0.7)
                response = llm.invoke([HumanMessage(content=question)])
                return response.content
            except Exception as e:
                return f"Error: {str(e)}"

    # Route and process
    _detected_type = _classify_question(user_question.value)
    _route = question_type.value  # Use user selection

    if _route == "math":
        _response = _handle_math(user_question.value)
        _route_path = "Question → Classifier → **MATH ROUTE** → Calculator → Response"
    else:
        _response = _handle_general(user_question.value, use_mock_routing.value)
        _route_path = "Question → Classifier → **GENERAL ROUTE** → LLM → Response"

    # Diagram
    _diagram = mo.md(f"""
    ```mermaid
    graph LR
        A[Input Question] --> B[Classify]
        B -->|Math| C[Calculator]
        B -->|General| D[LLM]
        C --> E[Response]
        D --> E
        style {"C" if _route == "math" else "D"} fill:#90EE90
    ```
    """)

    _content = mo.vstack([
        mo.md("## Conditional Routing: Intelligent Decision Making"),
        mo.md("_Route execution dynamically based on input classification._"),
        mo.md(""),
        _diagram,
        mo.md(""),
        mo.md("""
        ### Router Pattern

        ```python
        def router(state: MyState) -> str:
            \"\"\"Decide which node to execute next.\"\"\"
            question_type = classify(state["question"])

            if question_type == "math":
                return "calculator"
            else:
                return "llm"

        # Add conditional edge
        builder.add_conditional_edges(
            "classify",
            router,
            {
                "calculator": "calculator",
                "llm": "llm"
            }
        )
        ```

        ### Try It!
        """),
        mo.hstack([question_type, use_mock_routing]),
        user_question,
        mo.md(""),
        mo.callout(f"**Auto-detected type:** {_detected_type}", kind="info"),
        mo.md(f"**Route taken:** {_route_path}"),
        mo.md(""),
        mo.md(f"**Response:**"),
        mo.md(_response)
    ])

    return question_type, user_question, use_mock_routing, _content


@app.cell
def slide_8_tools(mo, tool):
    """Cell 8: Tools & Function Calling - Extending agent capabilities."""

    # Define calculator tool
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression. Use this for math calculations."""
        try:
            # Warning: eval is unsafe in production - for demo only!
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def string_reverser(text: str) -> str:
        """Reverse a string. Useful for text manipulation."""
        return text[::-1]

    # Interactive demo
    tool_choice = mo.ui.dropdown(
        options=["calculator", "string_reverser"],
        value="calculator",
        label="Select tool:"
    )

    tool_input = mo.ui.text(
        value="15 * 23" if tool_choice.value == "calculator" else "LangGraph",
        label="Tool input:"
    )

    # Execute tool
    if tool_choice.value == "calculator":
        _result = calculator.invoke({"expression": tool_input.value})
    else:
        _result = string_reverser.invoke({"text": tool_input.value})

    _content = mo.vstack([
        mo.md("## Tools: Extending Agent Capabilities"),
        mo.md("_Tools are functions that agents can invoke to perform specific tasks._"),
        mo.md(""),
        mo.md("""
        ### What Are Tools?

        **Tools** are decorated functions that:
        - Agents can automatically discover and invoke
        - Have clear descriptions (docstrings)
        - Accept typed parameters
        - Return results to the agent

        ### Defining a Tool

        ```python
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            \"\"\"Evaluate a mathematical expression.\"\"\"
            result = eval(expression)  # Demo only - unsafe in production!
            return f"Result: {result}"

        @tool
        def web_search(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            # Your search logic here
            return search_results
        ```

        ### Using Tools in Agents

        ```python
        from langchain_openai import ChatOpenAI

        # Create LLM with tool binding
        tools = [calculator, web_search]
        llm = ChatOpenAI(model="gpt-4").bind_tools(tools)

        def agent_node(state):
            messages = state["messages"]
            response = llm.invoke(messages)

            # Agent decides whether to call tool
            if response.tool_calls:
                # Execute tool calls
                results = execute_tools(response.tool_calls)
                return {"messages": results}
            return {"messages": [response]}
        ```

        ### Interactive Tool Demo

        Try invoking tools directly:
        """),
        mo.hstack([tool_choice, tool_input]),
        mo.md(""),
        mo.md(f"**Tool Output:**"),
        mo.code(_result, language="text"),
        mo.md(""),
        mo.callout(
            "**Key Concept**: In a real agent, the LLM decides *when* and *how* to use tools based on the user's request!",
            kind="success"
        )
    ])

    return tool_choice, tool_input, calculator, string_reverser, _content


@app.cell
def slide_9_react_pattern(mo, create_react_graph):
    """Cell 9: Cycles & Loops - ReAct Pattern."""

    # Interactive controls
    max_iterations = mo.ui.slider(
        start=1,
        stop=5,
        value=3,
        label="Max iterations:"
    )
    use_mock_react = mo.ui.checkbox(value=True, label="Mock mode")

    # Create ReAct graph
    react_graph = create_react_graph(max_iter=max_iterations.value, mock_mode=use_mock_react.value)

    # Diagram
    _diagram = mo.md("""
    ```mermaid
    graph LR
        A[Reason] --> B{Should Continue?}
        B -->|Yes| C[Use Tool]
        B -->|No| D[END]
        C --> E[Observe Result]
        E --> A
        style A fill:#FFE4B5
        style C fill:#90EE90
        style E fill:#ADD8E6
    ```
    """)

    _content = mo.vstack([
        mo.md("## Cycles & Loops: The ReAct Pattern"),
        mo.md("_Iterative reasoning with feedback loops - the foundation of agentic behavior._"),
        mo.md(""),
        _diagram,
        mo.md(""),
        mo.md("""
        ### What is ReAct?

        **ReAct** = **Reason** + **Act** + **Observe**

        A pattern where agents:
        1. **Reason** about what to do next
        2. **Act** by using tools or generating responses
        3. **Observe** results and loop back to reasoning
        4. Repeat until task is complete

        ### Why Use Cycles?

        Cycles enable:
        - **Iterative refinement**: Try, evaluate, improve
        - **Multi-step reasoning**: Break complex tasks into steps
        - **Tool use**: Call tools, see results, decide next action
        - **Error recovery**: Detect failures and try alternatives

        ### ReAct Implementation

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
        builder.add_node("reason", reason_node)
        builder.add_node("use_tool", tool_node)

        builder.add_conditional_edges("reason", should_continue, {
            "reason": "reason",      # Loop back!
            "use_tool": "use_tool",
            "finish": END
        })
        builder.add_edge("use_tool", "reason")  # Cycle back to reasoning
        ```

        ### Configure and Visualize
        """),
        max_iterations,
        use_mock_react,
        mo.md(""),
        mo.callout(
            f"Graph configured with max {max_iterations.value} iterations. "
            f"The agent will reason → act → observe → reason until complete!",
            kind="info"
        ),
        mo.md(""),
        mo.md("""
        ### Real-World ReAct Example

        **User**: "What's the weather in Paris and what should I wear?"

        **Iteration 1:**
        - Reason: "I need weather data for Paris"
        - Act: Call `get_weather("Paris")` tool
        - Observe: "Temperature: 18°C, Rainy"

        **Iteration 2:**
        - Reason: "Based on 18°C and rain, I can suggest clothing"
        - Act: Generate response
        - Result: "Wear a light jacket and bring an umbrella!"
        """)
    ])

    return max_iterations, use_mock_react, react_graph, _content


@app.cell
def slide_10_pipeline_architecture(mo):
    """Cell 10: Data Pipeline Architecture - Real-world example overview."""

    _diagram = mo.md("""
    ```mermaid
    graph LR
        A[Load Data] --> B[Analyze]
        B --> C[Generate Insights]
        C --> D[Visualize]
        D --> E[END]
        style A fill:#FFE4B5
        style B fill:#ADD8E6
        style C fill:#90EE90
        style D fill:#DDA0DD
    ```
    """)

    _content = mo.vstack([
        mo.md("## Real-World Example: Data Analysis Pipeline"),
        mo.md("_A complete workflow that loads data, analyzes it, generates AI insights, and visualizes results._"),
        mo.md(""),
        _diagram,
        mo.md(""),
        mo.md("""
        ### Pipeline Overview

        This real-world example demonstrates a **4-node data analysis pipeline** that combines:
        - Data processing (pandas)
        - Statistical analysis
        - AI-powered insights (LLM)
        - Interactive visualizations (plotly)

        ### Pipeline Stages

        #### 1. **Load Data** Node
        - Reads CSV file (sample or uploaded)
        - Validates data structure
        - Returns DataFrame in state

        ```python
        def load_data(state):
            df = pd.read_csv("data.csv")
            return {"data": df}
        ```

        #### 2. **Analyze** Node
        - Calculates descriptive statistics
        - Identifies trends and patterns
        - Aggregates by product, region, time

        ```python
        def analyze(state):
            df = state["data"]
            stats = {
                "total_revenue": df["revenue"].sum(),
                "top_product": df.groupby("product")["revenue"].sum().idxmax(),
                "avg_revenue": df["revenue"].mean(),
                # ... more stats
            }
            return {"stats": stats}
        ```

        #### 3. **Generate Insights** Node
        - LLM interprets statistics
        - Provides natural language insights
        - Generates actionable recommendations

        ```python
        def generate_insights(state):
            stats = state["stats"]
            llm = ChatOpenAI(model="gpt-4")
            prompt = f"Analyze these stats and provide insights: {stats}"
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"insights": response.content}
        ```

        #### 4. **Visualize** Node
        - Creates interactive charts
        - Shows revenue by product
        - Regional distribution
        - Time series trends

        ```python
        def visualize(state):
            df = state["data"]
            fig = px.bar(df.groupby("product")["revenue"].sum())
            return {"chart": fig}
        ```

        ### State Definition

        ```python
        class DataPipelineState(TypedDict):
            data: pd.DataFrame | None        # Raw data
            stats: dict | None                # Calculated statistics
            insights: str                     # AI-generated insights
            chart_data: dict | None          # Visualization data
        ```

        ### Why This Pattern?

        This pipeline demonstrates:
        - **Sequential processing**: Each node builds on previous results
        - **State accumulation**: Data flows through the pipeline
        - **Mixed processing**: Combines traditional code and AI
        - **Real-world applicability**: Template for data workflows

        ---

        **Next slide**: See the full implementation in action with interactive controls!
        """)
    ])

    return _content,


@app.cell
def slide_11_pipeline_implementation(mo, create_data_pipeline, os, create_revenue_chart, create_product_comparison, create_region_chart, create_stats_table, pd):
    """Cell 11: Data Pipeline Implementation - Full working demo."""

    # Interactive controls
    use_sample_data = mo.ui.checkbox(value=True, label="Use sample data")
    use_mock_insights = mo.ui.checkbox(value=True, label="Mock mode (no API calls)")

    # Get sample data path
    sample_path = os.path.join(os.path.dirname(__file__), "examples", "sample_data.csv")

    # Create and execute pipeline
    pipeline = create_data_pipeline(mock_mode=use_mock_insights.value, sample_data_path=sample_path if use_sample_data.value else None)

    try:
        _result = pipeline.invoke({})

        # Extract results
        _df = _result.get("data")
        _stats = _result.get("stats", {})
        _insights = _result.get("insights", "No insights generated")
        _chart_data = _result.get("chart_data", {})

        # Create visualizations
        if _df is not None and not _df.empty:
            _revenue_chart = create_revenue_chart(_df, chart_type="bar")
            _product_chart = create_product_comparison(_df)
            _region_chart = create_region_chart(_df)
            _stats_table = create_stats_table(_stats)
        else:
            _revenue_chart = None
            _product_chart = None
            _region_chart = None
            _stats_table = "_No data available_"

    except Exception as e:
        _result = {"error": str(e)}
        _df = None
        _stats = {}
        _insights = f"Error executing pipeline: {str(e)}"
        _revenue_chart = None
        _product_chart = None
        _region_chart = None
        _stats_table = f"_Error: {str(e)}_"

    _content = mo.vstack([
        mo.md("## Data Analysis Pipeline: Live Demo"),
        mo.md("_Full implementation with interactive controls - see the entire workflow in action!_"),
        mo.md(""),
        mo.hstack([use_sample_data, use_mock_insights]),
        mo.md(""),

        # Data overview
        mo.md("### 📊 Data Overview"),
        mo.md(f"**Records loaded:** {len(_df) if _df is not None else 0}") if _df is not None else mo.md("_No data loaded_"),
        mo.md(""),

        # Statistics
        mo.md("### 📈 Statistical Analysis"),
        mo.md(_stats_table),
        mo.md(""),

        # AI Insights
        mo.md("### 🤖 AI-Generated Insights"),
        mo.md(_insights),
        mo.md(""),

        # Visualizations
        mo.md("### 📉 Interactive Visualizations"),
        mo.ui.tabs({
            "Revenue by Product": mo.ui.plotly(_revenue_chart) if _revenue_chart else mo.md("_No chart data_"),
            "Product Comparison": mo.ui.plotly(_product_chart) if _product_chart else mo.md("_No chart data_"),
            "Regional Distribution": mo.ui.plotly(_region_chart) if _region_chart else mo.md("_No chart data_"),
        }),
        mo.md(""),

        mo.callout(
            f"{'✅ Pipeline executed successfully in mock mode!' if use_mock_insights.value else '🌐 Pipeline executed with real LLM insights!'}",
            kind="success"
        ),

        mo.md(""),
        mo.md("""
        ### How It Works

        The pipeline executed these steps:

        1. **Load Data**: Read CSV file (sample_data.csv with 90 transactions)
        2. **Analyze**: Calculated revenue stats, top products, regional performance
        3. **Generate Insights**: LLM analyzed patterns and provided recommendations
        4. **Visualize**: Created interactive charts for exploration

        ### Try It Yourself!

        - Toggle mock mode off and add your OpenAI API key to `.env` for real AI insights
        - Upload your own CSV with columns: date, product, region, revenue, quantity
        - Experiment with different datasets to see how the pipeline adapts

        ### Key Takeaways

        This pipeline demonstrates:
        - **State management**: Data flows through nodes via shared state
        - **Mixed processing**: Combines pandas, LLMs, and visualization
        - **Modularity**: Each node has a single, clear responsibility
        - **Reusability**: Template can be adapted for various data workflows
        """)
    ])

    return use_sample_data, use_mock_insights, pipeline, _content


@app.cell
def slide_12_langsmith(mo):
    """Cell 12: Observability with LangSmith - Tracing and debugging."""

    _content = mo.vstack([
        mo.md("## Observability: LangSmith Tracing"),
        mo.md("_Debug, optimize, and understand your LangGraph applications with comprehensive tracing._"),
        mo.md(""),
        mo.md("""
        ### Why Observability Matters

        As your LangGraph applications grow complex, you need visibility into:
        - **Execution flow**: Which nodes ran and in what order?
        - **Performance**: Where are the bottlenecks?
        - **Costs**: How many tokens are you using?
        - **Errors**: What failed and why?
        - **Quality**: Are outputs meeting expectations?

        ### LangSmith Configuration

        LangSmith tracing is already configured in this presentation via `.env`:

        ```bash
        # Enable tracing
        LANGCHAIN_TRACING_V2=true

        # Your API key
        LANGCHAIN_API_KEY=lsv2_your_key_here

        # Project name for organization
        LANGCHAIN_PROJECT=langgraph-presentation
        ```

        ### What You Get

        #### 1. **Execution Traces**
        - Hierarchical view of all nodes and LLM calls
        - Input/output for each step
        - Timing information
        - Parent-child relationships

        #### 2. **Token Usage Tracking**
        - Count tokens per LLM call
        - Aggregate costs across runs
        - Identify expensive operations
        - Optimize prompts

        #### 3. **Error Debugging**
        - Stack traces for failures
        - State at time of error
        - Retry history
        - Root cause analysis

        #### 4. **Performance Optimization**
        - Identify slow nodes
        - Parallel vs sequential execution
        - Caching opportunities
        - Bottleneck detection

        ### Code Example

        ```python
        import os
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        # That's it! Tracing is now automatic
        # Every LangGraph execution is traced to LangSmith

        # Run your graph
        result = my_graph.invoke({"input": "Hello"})

        # View traces at: https://smith.langchain.com
        ```

        ### Viewing Traces

        1. **Visit**: [https://smith.langchain.com](https://smith.langchain.com)
        2. **Select your project**: `langgraph-presentation`
        3. **Browse traces**: See all executions
        4. **Drill down**: Click any trace to see details

        ### Advanced Features

        #### Evaluation
        ```python
        from langsmith import Client

        client = Client()

        # Create evaluation datasets
        client.create_dataset("test-cases", ...)

        # Run evaluations
        results = client.evaluate(my_graph, dataset="test-cases")
        ```

        #### Annotations
        ```python
        # Add custom metadata to traces
        with tracing_v2_enabled(
            project_name="my-project",
            tags=["production", "v2"],
            metadata={"user_id": "123"}
        ):
            result = my_graph.invoke(input_data)
        ```

        ### Best Practices

        1. **Use descriptive project names**: Organize by environment (dev, staging, prod)
        2. **Add tags**: Categorize runs (experiment, baseline, user-test)
        3. **Review regularly**: Check traces after deployments
        4. **Set alerts**: Monitor for errors and high costs
        5. **Evaluate systematically**: Use datasets to track quality

        ### Integration with This Presentation

        This marimo deck is already configured with LangSmith tracing:
        - All LLM calls in "real mode" are traced automatically
        - Check your LangSmith dashboard after running examples
        - No code changes needed - just enable in `.env`

        """),
        mo.callout(
            "🎯 **Pro Tip**: Start tracing from day one - it's invaluable for debugging complex graphs!",
            kind="success"
        )
    ])

    return _content,


@app.cell
def slide_13_advanced_patterns(mo):
    """Cell 13: Advanced Patterns - Quick overview of next-level concepts."""

    _content = mo.vstack([
        mo.md("## Advanced Patterns & Next Steps"),
        mo.md("_Beyond the basics - powerful patterns for production applications._"),
        mo.md(""),
        mo.md("""
        ### 1. Subgraphs: Modularity and Reusability

        **Nest entire graphs within nodes** for complex workflows.

        ```python
        # Create a reusable subgraph
        def create_validation_subgraph():
            builder = StateGraph(ValidationState)
            builder.add_node("check_format", check_format)
            builder.add_node("check_content", check_content)
            builder.add_edge("check_format", "check_content")
            return builder.compile()

        # Use subgraph as a node
        main_builder = StateGraph(MainState)
        main_builder.add_node("validate", create_validation_subgraph())
        ```

        **Use cases**: Auth flows, validation pipelines, retry logic

        ---

        ### 2. Multi-Agent Orchestration

        **Multiple specialized agents** collaborating on complex tasks.

        ```python
        # Researcher agent
        researcher = create_agent(
            tools=[web_search, database_query],
            system_prompt="You are a research specialist"
        )

        # Writer agent
        writer = create_agent(
            tools=[text_editor, formatter],
            system_prompt="You are a content writer"
        )

        # Orchestrator decides which agent to use
        def route_to_agent(state):
            if state["task_type"] == "research":
                return "researcher"
            return "writer"
        ```

        **Use cases**: Customer support (routing), content creation, code generation

        ---

        ### 3. Human-in-the-Loop

        **Pause execution** for human review and approval.

        ```python
        from langgraph.checkpoint import MemorySaver

        # Add checkpointer for persistence
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        # Add interrupt before critical actions
        builder.add_node("needs_approval", review_action)
        builder.add_edge("needs_approval", "execute", interrupt=True)

        # Run until interrupt
        config = {"configurable": {"thread_id": "user-123"}}
        result = graph.invoke(input_data, config)

        # Resume after human approval
        result = graph.invoke(None, config)  # Continues from checkpoint
        ```

        **Use cases**: Content moderation, financial approvals, sensitive operations

        ---

        ### 4. Persistence and Checkpointing

        **Save and restore** graph state across sessions.

        ```python
        from langgraph.checkpoint import SqliteSaver

        # Use database checkpointer
        checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        graph = builder.compile(checkpointer=checkpointer)

        # State is automatically saved at each step
        config = {"configurable": {"thread_id": "conversation-1"}}

        # Run can be resumed any time
        result = graph.invoke(input_data, config)
        ```

        **Use cases**: Long-running workflows, chatbots, resumable processes

        ---

        ### 5. Error Handling and Retries

        **Robust workflows** with automatic retry logic.

        ```python
        def node_with_retry(state):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = risky_operation(state)
                    return {"result": result, "error": None}
                except Exception as e:
                    if attempt == max_retries - 1:
                        return {"result": None, "error": str(e)}
                    continue

        # Route based on error
        def handle_error(state):
            if state["error"]:
                return "retry"
            return "success"

        builder.add_conditional_edges("process", handle_error, {
            "retry": "process",  # Cycle back
            "success": "next_step"
        })
        ```

        **Use cases**: API calls, external integrations, unreliable operations

        ---

        ### 6. Parallel Execution

        **Run multiple nodes simultaneously** for efficiency.

        ```python
        # These nodes can run in parallel
        builder.add_node("fetch_weather", fetch_weather)
        builder.add_node("fetch_news", fetch_news)
        builder.add_node("fetch_stocks", fetch_stocks)

        # All start from same source
        builder.add_edge("start", "fetch_weather")
        builder.add_edge("start", "fetch_news")
        builder.add_edge("start", "fetch_stocks")

        # All converge to aggregator
        builder.add_edge("fetch_weather", "aggregate")
        builder.add_edge("fetch_news", "aggregate")
        builder.add_edge("fetch_stocks", "aggregate")
        ```

        **Use cases**: Data gathering, parallel API calls, fan-out/fan-in

        ---

        ### 7. Streaming and Real-Time Updates

        **Stream partial results** as graph executes.

        ```python
        # Stream events in real-time
        for event in graph.stream(input_data):
            node_name = event["node"]
            state_update = event["state"]
            print(f"Node {node_name}: {state_update}")
        ```

        **Use cases**: Live dashboards, progressive loading, chat applications

        ---

        ### Resources for Learning More

        #### Official Documentation
        - [LangGraph Docs](https://langchain-ai.github.io/langgraph/) - Comprehensive guides
        - [LangChain Docs](https://python.langchain.com/) - Core concepts
        - [LangSmith](https://smith.langchain.com/) - Observability platform

        #### Examples and Tutorials
        - [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples) - Official examples
        - [Real Python Tutorial](https://realpython.com/langgraph-python/) - Beginner-friendly guide
        - [DataCamp Tutorial](https://www.datacamp.com/tutorial/langgraph-tutorial) - Hands-on exercises

        #### Community
        - [Discord](https://discord.gg/langchain) - Get help and share ideas
        - [GitHub Discussions](https://github.com/langchain-ai/langgraph/discussions) - Technical questions
        - [Twitter/X](https://twitter.com/langchainai) - Updates and announcements

        """)
    ])

    return _content,


@app.cell
def slide_14_summary(mo):
    """Cell 14: Summary & Next Steps - Wrap up and resources."""

    _content = mo.vstack([
        mo.md("# 🎯 Key Takeaways"),
        mo.md(""),
        mo.md("""
        ## What We Learned

        ### 1. **Graphs > Chains**
        - Cycles enable iterative refinement
        - Conditional branching creates adaptive workflows
        - State management provides persistent memory
        - Complex patterns that linear chains can't achieve

        ### 2. **Core Concepts Mastery**
        - **State**: Shared memory as TypedDict or MessagesState
        - **Nodes**: Functions that transform state
        - **Edges**: Control flow (fixed or conditional)
        - **Graphs**: Compile nodes and edges into executable workflows

        ### 3. **Building Intelligent Agents**
        - Integrate LLMs seamlessly within nodes
        - Use tools to extend capabilities
        - Implement ReAct pattern for reasoning loops
        - Route dynamically based on input

        ### 4. **Production Patterns**
        - Data pipelines with mixed processing
        - LangSmith tracing for observability
        - Error handling and retries
        - Multi-agent orchestration

        ---

        ## 🚀 Next Steps

        ### Immediate Actions

        1. **Run This Presentation Locally**
        ```bash
        cd langgraph
        uv sync
        cp .env.example .env
        # Add your OPENAI_API_KEY to .env
        uv run marimo run langgraph_deck.py
        ```

        2. **Experiment with Examples**
        - Modify the data pipeline with your own CSV
        - Try different question types in routing
        - Adjust max iterations in ReAct pattern
        - Toggle mock mode on/off

        3. **Build Your First Graph**
        - Start with a simple 3-node linear graph
        - Add conditional routing
        - Integrate an LLM
        - Add tool use

        ### Learning Path

        #### **Week 1: Foundations**
        - Build simple linear graphs
        - Practice state management
        - Implement fixed and conditional edges

        #### **Week 2: Intelligence**
        - Add LLM integration
        - Create routing logic
        - Implement tool calling

        #### **Week 3: Advanced Patterns**
        - Build ReAct loops
        - Create multi-agent systems
        - Add human-in-the-loop

        #### **Week 4: Production**
        - Implement error handling
        - Add checkpointing and persistence
        - Set up LangSmith monitoring
        - Deploy your first application

        ---

        ## 📚 Essential Resources

        ### Official Documentation
        - **LangGraph**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
        - **LangChain**: [python.langchain.com](https://python.langchain.com/)
        - **LangSmith**: [smith.langchain.com](https://smith.langchain.com/)
        - **Marimo**: [docs.marimo.io](https://docs.marimo.io/)

        ### Tutorials & Guides
        - [Real Python LangGraph Tutorial](https://realpython.com/langgraph-python/)
        - [DataCamp LangGraph Guide](https://www.datacamp.com/tutorial/langgraph-tutorial)
        - [Official Examples Repository](https://github.com/langchain-ai/langgraph/tree/main/examples)

        ### Community & Support
        - **Discord**: [discord.gg/langchain](https://discord.gg/langchain)
        - **GitHub**: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
        - **Twitter/X**: [@langchainai](https://twitter.com/langchainai)

        ---

        ## 💡 Project Ideas

        Ready to build? Try these projects:

        ### Beginner
        1. **FAQ Bot**: Route questions to appropriate response templates
        2. **Data Analyzer**: Upload CSV, get insights and charts
        3. **Task Classifier**: Categorize and route tasks to handlers

        ### Intermediate
        4. **Research Assistant**: Search web, summarize findings, cite sources
        5. **Code Reviewer**: Analyze code, identify issues, suggest fixes
        6. **Content Generator**: Research topic, create outline, write sections

        ### Advanced
        7. **Multi-Agent Customer Support**: Route to specialists, escalate when needed
        8. **Automated Testing Pipeline**: Generate tests, run them, analyze results
        9. **Data Science Workflow**: Load → clean → analyze → visualize → report

        ---

        ## 🙏 Thank You!

        You now have the foundation to build sophisticated, stateful AI applications with LangGraph.

        **Remember**:
        - Start simple, add complexity gradually
        - Use mock mode while developing
        - Trace everything with LangSmith
        - Iterate and refine based on results

        ### This Presentation

        - **Source**: Available in this repository
        - **License**: MIT - use it freely!
        - **Feedback**: Open an issue or PR

        ---

        ## Ready to Build?

        ```python
        from langgraph.graph import StateGraph, END

        # Your journey starts here
        builder = StateGraph(YourState)
        builder.add_node("start", your_first_node)
        builder.set_entry_point("start")
        builder.add_edge("start", END)

        graph = builder.compile()
        result = graph.invoke({"input": "Hello, LangGraph!"})

        print(result)  # Your first LangGraph execution!
        ```

        **Happy building! 🚀**
        """)
    ])

    return _content,


if __name__ == "__main__":
    app.run()
