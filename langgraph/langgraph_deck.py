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

    from graph_helpers import (
        create_linear_graph,
        create_react_graph,
        create_multi_tool_graph,
        create_llm_tool_agent,
        create_multi_step_workflow,
        create_multi_agent_graph
    )

    # Load environment variables silently
    load_dotenv()

    # Configure LangSmith tracing (happens silently in background)
    # No output needed - just configuration
    return (
        mo, os, load_dotenv, StateGraph, END, HumanMessage, AIMessage,
        ChatOpenAI, tool, TypedDict, Annotated, operator, pd, px, go, io,
        create_linear_graph, create_react_graph,
        create_multi_tool_graph, create_llm_tool_agent,
        create_multi_step_workflow, create_multi_agent_graph
    )


@app.cell
def slide_1_introduction(mo):
    """Cell 1: Introduction - What is LangGraph and why it matters."""
    mo.md("""
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
    return node_input,


@app.cell
def slide_3_nodes_content(mo, node_input, TypedDict):
    """Cell 3b: Nodes - Display node processing."""

    class ExampleState(TypedDict):
        input: str
        output: str

    def process_node(state: ExampleState) -> dict:
        input_val = state.get("input", "")
        processed = input_val.upper() + " - Processed by node!"
        return {"output": processed}

    state_val = {"input": node_input.value}
    result_val = process_node(state_val)

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

        ### Try It Live!
        Change the input below and see how the node processes it:
        """),
        mo.md(""),
        node_input,
        mo.md(""),
        mo.md(f"""
        **Input:** `{node_input.value}`

        **Output:** `{result_val['output']}`
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

    fixed_ex = mo.md("""
    ### Fixed Edges
    **Always go to the same next node** - simple sequential flow.

    ```python
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")
    ```

    **Use Cases:** Simple pipelines, sequential processing
    """)

    conditional_ex = mo.md("""
    ### Conditional Edges
    **Route based on state** - dynamic decision making!

    ```python
    def router(state):
        if state["type"] == "math":
            return "calculator"
        return "llm"

    builder.add_conditional_edges("classify", router, {
        "calculator": "calculator",
        "llm": "llm"
    })
    ```

    **Use Cases:** Intelligent routing, decision trees, adaptive workflows
    """)

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
    return graph_input,


@app.cell
def slide_5_graph_content(mo, graph_input, create_linear_graph):
    """Cell 5b: First Graph - Execute and display."""

    linear_g = create_linear_graph()

    try:
        result_g = linear_g.invoke({"input": graph_input.value})
    except Exception as e:
        result_g = {"error": str(e)}

    mo.vstack([
        mo.md("## Your First LangGraph: Linear Workflow"),
        mo.md("_Let's build a simple 3-node sequential graph!_"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            START([START]) --> Node1[node1]
            Node1 --> Node2[node2]
            Node2 --> Node3[node3]
            Node3 --> END([END])
        """),
        mo.md("### Try It Live!"),
        graph_input,
        mo.md(""),
        mo.md(f"""
        **Execution Result:**
        - Input: `{graph_input.value}`
        - Step 1: `{result_g.get('step1', 'N/A')}`
        - Step 2: `{result_g.get('step2', 'N/A')}`
        - Output: `{result_g.get('output', 'N/A')}`
        """)
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
        mo.hstack([mock_mode_llm, mo.md(f"_{'Mock mode' if mock_mode_llm.value else 'Real OpenAI API'}_")]),
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
    else:
        response_routing = handle_general_q(user_question_ui.value, use_mock_routing_ui.value)

    mo.vstack([
        mo.md("## Conditional Routing: Intelligent Decision Making"),
        mo.md("_Route execution dynamically based on input classification._"),
        mo.md(""),
        mo.hstack([question_type_ui, use_mock_routing_ui]),
        user_question_ui,
        mo.md(""),
        mo.md(f"**Response:** {response_routing}")
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
        mo.hstack([tool_choice_ui, tool_input_ui]),
        mo.md(""),
        mo.md(f"**Tool Output:** `{tool_result}`"),
        mo.callout(
            "In real agents, the LLM decides when to use tools!",
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

    mo.vstack([
        mo.md("## Cycles & Loops: The ReAct Pattern"),
        mo.md("_Iterative reasoning with feedback loops._"),
        mo.md(""),
        mo.md("**ReAct** = **Reason** + **Act** + **Observe**"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            A[Reason] --> B{Continue?}
            B -->|Yes| C[Use Tool]
            B -->|No| D[END]
            C --> E[Observe]
            E --> A
        """),
        mo.md(""),
        mo.md("Cycles enable iterative refinement and multi-step reasoning."),
        mo.md(""),
        max_iterations_ui,
        use_mock_react_ui,
        mo.callout(
            f"Graph configured with max {max_iterations_ui.value} iterations.",
            kind="info"
        )
    ])


@app.cell
def slide_10_multi_tool_ui(mo):
    """Cell 10a: Enhanced Multi-Tool Routing - Create UI elements."""
    tool_selector = mo.ui.dropdown(
        options=["calculator", "string_reverser", "word_counter", "text_transformer"],
        value="calculator",
        label="Select tool:"
    )
    tool_input_field = mo.ui.text(
        value="25 * 4",
        label="Input:"
    )
    return tool_selector, tool_input_field


@app.cell
def slide_10_multi_tool_content(mo, tool_selector, tool_input_field, create_multi_tool_graph):
    """Cell 10b: Enhanced Multi-Tool Routing - Execute and display."""

    multi_tool_graph = create_multi_tool_graph()

    try:
        result_multi_tool = multi_tool_graph.invoke({
            "input": tool_input_field.value,
            "selected_tool": tool_selector.value
        })
        output = result_multi_tool.get("output", "No output")
    except Exception as e:
        output = f"Error: {str(e)}"

    return mo.vstack([
        mo.md("## Example 1: Enhanced Multi-Tool Routing"),
        mo.md("_Conditional routing with 4 different tools - the graph routes to the selected tool._"),
        mo.md(""),
        mo.md("### Graph Structure"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            START([START]) --> Router{Router}
            Router -->|calculator| Calc[Calculator]
            Router -->|string_reverser| Rev[String Reverser]
            Router -->|word_counter| Count[Word Counter]
            Router -->|text_transformer| Trans[Text Transformer]
            Calc --> END([END])
            Rev --> END
            Count --> END
            Trans --> END

            style Router fill:#ffd700
            style Calc fill:#90EE90
            style Rev fill:#90EE90
            style Count fill:#90EE90
            style Trans fill:#90EE90
        """),
        mo.md(""),
        mo.md("### Try It!"),
        mo.hstack([tool_selector, tool_input_field]),
        mo.md(""),
        mo.md(f"**Result:** `{output}`"),
        mo.md(""),
        mo.md("""
        ### Key Code

        ```python
        # Define tools
        @tool
        def calculator(expression: str) -> str:
            \"\"\"Evaluate a mathematical expression.\"\"\"
            return str(eval(expression))

        @tool
        def string_reverser(text: str) -> str:
            \"\"\"Reverse a string.\"\"\"
            return text[::-1]

        @tool
        def word_counter(text: str) -> str:
            \"\"\"Count words and characters.\"\"\"
            words = len(text.split())
            chars = len(text)
            return f"Words: {words}, Characters: {chars}"

        @tool
        def text_transformer(text: str) -> str:
            \"\"\"Transform text case.\"\"\"
            return f"Upper: {text.upper()} | Lower: {text.lower()} | Title: {text.title()}"

        # Router function
        def router(state) -> str:
            return state["selected_tool"]

        # Build graph
        builder.add_conditional_edges("router", router, {
            "calculator": "calculator_node",
            "string_reverser": "string_reverser_node",
            "word_counter": "word_counter_node",
            "text_transformer": "text_transformer_node"
        })
        ```
        """)
    ])


@app.cell
def slide_11_llm_agent_ui(mo):
    """Cell 11a: LLM Agent with Dynamic Tool Selection - Create UI elements."""
    agent_question = mo.ui.text_area(
        value="What is 127 * 45?",
        label="Ask a question (the LLM will choose the right tool):",
        rows=3
    )
    use_mock_agent = mo.ui.checkbox(value=True, label="Mock mode (no API)")
    return agent_question, use_mock_agent


@app.cell
def slide_11_llm_agent_content(mo, agent_question, use_mock_agent, create_llm_tool_agent):
    """Cell 11b: LLM Agent with Dynamic Tool Selection - Execute and display."""

    llm_agent_graph = create_llm_tool_agent(mock_mode=use_mock_agent.value)

    try:
        result_llm_agent = llm_agent_graph.invoke({"question": agent_question.value})
        tool_used = result_llm_agent.get("tool_to_use", "None")
        tool_result_agent = result_llm_agent.get("tool_result", "")
        final_answer = result_llm_agent.get("final_answer", "No answer")
    except Exception as e:
        tool_used = "Error"
        tool_result_agent = ""
        final_answer = f"Error: {str(e)}"

    return mo.vstack([
        mo.md("## Example 2: LLM Agent with Dynamic Tool Selection"),
        mo.md("_The LLM analyzes your question and automatically chooses the appropriate tool._"),
        mo.md(""),
        mo.md("### Graph Structure"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            START([START]) --> Classify[LLM Classify]
            Classify --> Decision{Should Use Tool?}
            Decision -->|Yes| Tool[Execute Tool]
            Decision -->|No| Direct[Direct Response]
            Tool --> Format[Format Response]
            Format --> END([END])
            Direct --> END

            style Classify fill:#87CEEB
            style Decision fill:#ffd700
            style Tool fill:#90EE90
            style Direct fill:#90EE90
            style Format fill:#FFB6C1
        """),
        mo.md(""),
        mo.md("### Try It!"),
        mo.hstack([use_mock_agent]),
        agent_question,
        mo.md(""),
        mo.md(f"**LLM Selected Tool:** `{tool_used}`"),
        mo.md(f"**Tool Result:** `{tool_result_agent}`") if tool_result_agent else mo.md(""),
        mo.md(f"**Final Answer:** {final_answer}"),
        mo.md(""),
        mo.callout(
            "💡 Try different questions: math ('What is 50/2?'), text ('Reverse hello'), counting ('Count words in: the quick brown fox'), or transformations ('Make HELLO lowercase')",
            kind="info"
        ),
        mo.md(""),
        mo.md("""
        ### Key Code

        ```python
        def llm_classify_node(state):
            \"\"\"LLM decides which tool to use.\"\"\"
            llm = ChatOpenAI(model="gpt-4")
            prompt = f\"\"\"Analyze this question and choose the best tool:

Question: {state['question']}

Tools available:
- calculator: for math calculations
- string_reverser: to reverse text
- word_counter: to count words/characters
- text_transformer: to change text case
- none: if no tool needed

Respond with ONLY the tool name.\"\"\"

            response = llm.invoke([HumanMessage(content=prompt)])
            tool_name = response.content.strip().lower()
            return {"tool_to_use": tool_name}

        def should_use_tool(state) -> str:
            \"\"\"Route based on LLM's decision.\"\"\"
            valid_tools = ["calculator", "string_reverser", "word_counter", "text_transformer"]
            if state["tool_to_use"] in valid_tools:
                return "use_tool"
            return "direct_response"

        # Conditional routing based on LLM decision
        builder.add_conditional_edges("classify", should_use_tool, {
            "use_tool": "execute_tool",
            "direct_response": "format_response"
        })
        ```
        """)
    ])


@app.cell
def slide_12_multi_step_ui(mo):
    """Cell 12a: Multi-Step Reasoning - Create UI elements."""
    complex_question = mo.ui.text_area(
        value="Explain quantum computing and provide 3 real-world use cases",
        label="Ask a complex question:",
        rows=3
    )
    max_iterations = mo.ui.slider(start=1, stop=5, value=3, label="Max iterations:")
    use_mock_multi_step = mo.ui.checkbox(value=True, label="Mock mode (no API)")
    return complex_question, max_iterations, use_mock_multi_step


@app.cell
def slide_12_multi_step_content(mo, complex_question, max_iterations, use_mock_multi_step, create_multi_step_workflow):
    """Cell 12b: Multi-Step Reasoning - Execute and display."""

    multi_step_graph = create_multi_step_workflow(
        mock_mode=use_mock_multi_step.value,
        max_iter=max_iterations.value
    )

    try:
        result_multi_step = multi_step_graph.invoke({"question": complex_question.value})
        plan = result_multi_step.get("plan", "No plan")
        research_notes = result_multi_step.get("research_notes", [])
        synthesis = result_multi_step.get("synthesis", "No synthesis")
        iteration_count = result_multi_step.get("iteration_count", 0)
    except Exception as e:
        plan = f"Error: {str(e)}"
        research_notes = []
        synthesis = ""
        iteration_count = 0

    return mo.vstack([
        mo.md("## Example 3: Multi-Step Reasoning Workflow"),
        mo.md("_Complex questions are broken down into steps, researched iteratively, then synthesized._"),
        mo.md(""),
        mo.md("### Graph Structure"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            START([START]) --> Plan[Plan Steps]
            Plan --> Research[Research]
            Research --> Synthesize[Synthesize]
            Synthesize --> Decision{Continue?}
            Decision -->|Incomplete| Research
            Decision -->|Complete/Max| END([END])

            style Plan fill:#87CEEB
            style Research fill:#90EE90
            style Synthesize fill:#FFB6C1
            style Decision fill:#ffd700
        """),
        mo.md(""),
        mo.md("### Try It!"),
        mo.hstack([max_iterations, use_mock_multi_step]),
        complex_question,
        mo.md(""),
        mo.md(f"**Iterations:** {iteration_count} / {max_iterations.value}"),
        mo.md(""),
        mo.md("### 📋 Plan"),
        mo.md(plan),
        mo.md(""),
        mo.md("### 🔬 Research Notes") if research_notes else mo.md(""),
        mo.vstack([mo.md(f"{i+1}. {note}") for i, note in enumerate(research_notes)]) if research_notes else mo.md(""),
        mo.md(""),
        mo.md("### 📝 Final Synthesis"),
        mo.md(synthesis),
        mo.md(""),
        mo.md("""
        ### Key Code

        ```python
        def plan_node(state):
            \"\"\"Break question into research steps.\"\"\"
            llm = ChatOpenAI(model="gpt-4")
            prompt = f\"\"\"Break this complex question into 3 specific research steps:

Question: {state['question']}

Return ONLY a numbered list of research steps.\"\"\"

            response = llm.invoke([HumanMessage(content=prompt)])
            return {"plan": response.content}

        def research_node(state):
            \"\"\"Research one step at a time.\"\"\"
            llm = ChatOpenAI(model="gpt-4")
            iteration = state["iteration_count"]
            steps = state["plan"].split("\\n")

            if iteration < len(steps):
                step = steps[iteration]
                prompt = f"Research this step: {step}"
                response = llm.invoke([HumanMessage(content=prompt)])
                return {
                    "research_notes": [response.content],
                    "iteration_count": iteration + 1
                }
            return {}

        def should_continue(state) -> str:
            \"\"\"Decide whether to continue research or finish.\"\"\"
            if state["iteration_count"] >= state["max_iterations"]:
                return "finish"
            if len(state["research_notes"]) >= 3:  # Completed all steps
                return "finish"
            return "research"

        # Cycle allows iterative refinement
        builder.add_conditional_edges("synthesize", should_continue, {
            "research": "research",
            "finish": END
        })
        ```
        """)
    ])


@app.cell
def slide_13_multi_agent_ui(mo):
    """Cell 13a: Multi-Agent Collaboration - Create UI elements."""
    topic_input = mo.ui.text(
        value="Benefits of using LangGraph for AI applications",
        label="Topic to write about:"
    )
    max_revisions = mo.ui.slider(start=1, stop=3, value=2, label="Max revisions:")
    use_mock_agents = mo.ui.checkbox(value=True, label="Mock mode (no API)")
    return topic_input, max_revisions, use_mock_agents


@app.cell
def slide_13_multi_agent_content(mo, topic_input, max_revisions, use_mock_agents, create_multi_agent_graph):
    """Cell 13b: Multi-Agent Collaboration - Execute and display."""

    multi_agent_graph = create_multi_agent_graph(
        mock_mode=use_mock_agents.value,
        max_revisions=max_revisions.value
    )

    try:
        result_multi_agent = multi_agent_graph.invoke({"topic": topic_input.value})
        draft = result_multi_agent.get("draft", "No draft")
        critic_feedback = result_multi_agent.get("critic_feedback", [])
        revision_count = result_multi_agent.get("revision_count", 0)
        approved = result_multi_agent.get("approved", False)
    except Exception as e:
        draft = f"Error: {str(e)}"
        critic_feedback = []
        revision_count = 0
        approved = False

    return mo.vstack([
        mo.md("## Example 4: Multi-Agent Collaboration"),
        mo.md("_Two LLMs with different roles (Writer & Critic) collaborate to produce quality content._"),
        mo.md(""),
        mo.md("### Graph Structure"),
        mo.md(""),
        mo.mermaid("""
        graph LR
            START([START]) --> Writer[Writer: Draft]
            Writer --> Critic[Critic: Review]
            Critic --> Decision{Approved?}
            Decision -->|No + Can Revise| Revise[Writer: Revise]
            Decision -->|Yes or Max Reached| END([END])
            Revise --> Critic

            style Writer fill:#87CEEB
            style Critic fill:#FFB6C1
            style Decision fill:#ffd700
            style Revise fill:#87CEEB
        """),
        mo.md(""),
        mo.md("### Try It!"),
        mo.hstack([max_revisions, use_mock_agents]),
        topic_input,
        mo.md(""),
        mo.md(f"**Status:** {'✅ Approved' if approved else f'🔄 Revisions: {revision_count}/{max_revisions.value}'}"),
        mo.md(""),
        mo.md("### ✍️ Current Draft"),
        mo.md(draft),
        mo.md(""),
        mo.md("### 📝 Critic Feedback History") if critic_feedback else mo.md(""),
        mo.vstack([
            mo.callout(
                f"**Round {i+1}:** {feedback}",
                kind="success" if "APPROVED" in feedback.upper() else "warn"
            )
            for i, feedback in enumerate(critic_feedback)
        ]) if critic_feedback else mo.md(""),
        mo.md(""),
        mo.md("""
        ### Key Code

        ```python
        def writer_draft_node(state):
            \"\"\"Writer agent creates content.\"\"\"
            llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Creative
            prompt = f\"\"\"You are a technical writer. Write a clear, concise explanation about:

Topic: {state['topic']}

Write 2-3 paragraphs.\"\"\"

            draft = llm.invoke([HumanMessage(content=prompt)]).content
            return {"draft": draft}

        def critic_review_node(state):
            \"\"\"Critic agent reviews and provides feedback.\"\"\"
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)  # Analytical
            prompt = f\"\"\"You are a critic reviewing technical content.

Draft: {state['draft']}

Provide either:
1. "APPROVED" if the draft is excellent
2. Specific feedback for improvement

Keep feedback concise.\"\"\"

            feedback = llm.invoke([HumanMessage(content=prompt)]).content
            approved = "APPROVED" in feedback.upper()

            return {
                "critic_feedback": [feedback],
                "approved": approved
            }

        def should_revise(state) -> str:
            \"\"\"Decide whether to revise or finish.\"\"\"
            if state["approved"]:
                return "finish"
            elif state["revision_count"] >= state["max_revisions"]:
                return "finish"
            return "revise"

        # Collaboration loop
        builder.add_conditional_edges("critic", should_revise, {
            "revise": "writer_revise",
            "finish": END
        })
        ```
        """)
    ])


@app.cell
def slide_14_langsmith(mo):
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
def slide_15_advanced(mo):
    """Cell 15: Advanced Patterns."""
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
def slide_16_summary(mo):
    """Cell 16: Summary & Next Steps."""
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
