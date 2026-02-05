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


class MultiToolState(TypedDict):
    """State for multi-tool routing example."""
    input: str
    selected_tool: str
    output: str


class LLMAgentState(TypedDict):
    """State for LLM agent with tool selection."""
    question: str
    tool_to_use: str
    tool_result: str
    final_answer: str


class MultiStepState(TypedDict):
    """State for multi-step reasoning workflow."""
    question: str
    plan: str
    research_notes: Annotated[list, operator.add]
    synthesis: str
    iteration_count: int
    max_iterations: int


class MultiAgentState(TypedDict):
    """State for multi-agent collaboration."""
    topic: str
    draft: str
    critic_feedback: Annotated[list, operator.add]
    revision_count: int
    max_revisions: int
    approved: bool


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


def create_multi_tool_graph():
    """
    Create a multi-tool routing graph with 4 different tools.

    Graph: router → [calculator | string_reverser | word_counter | text_transformer] → END
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

    @tool
    def string_reverser(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    @tool
    def word_counter(text: str) -> str:
        """Count words and characters in text."""
        words = len(text.split())
        chars = len(text)
        return f"Words: {words}, Characters: {chars}"

    @tool
    def text_transformer(text: str) -> str:
        """Transform text to different cases."""
        return f"Upper: {text.upper()} | Lower: {text.lower()} | Title: {text.title()}"

    def router_node(state: MultiToolState) -> dict:
        """Router passes through the selected tool."""
        return {}

    def calculator_node(state: MultiToolState) -> dict:
        """Execute calculator tool."""
        result = calculator.invoke({"expression": state["input"]})
        return {"output": result}

    def string_reverser_node(state: MultiToolState) -> dict:
        """Execute string reverser tool."""
        result = string_reverser.invoke({"text": state["input"]})
        return {"output": result}

    def word_counter_node(state: MultiToolState) -> dict:
        """Execute word counter tool."""
        result = word_counter.invoke({"text": state["input"]})
        return {"output": result}

    def text_transformer_node(state: MultiToolState) -> dict:
        """Execute text transformer tool."""
        result = text_transformer.invoke({"text": state["input"]})
        return {"output": result}

    def route_to_tool(state: MultiToolState) -> str:
        """Route based on selected tool."""
        return state["selected_tool"]

    # Build the graph
    builder = StateGraph(MultiToolState)
    builder.add_node("router", router_node)
    builder.add_node("calculator_node", calculator_node)
    builder.add_node("string_reverser_node", string_reverser_node)
    builder.add_node("word_counter_node", word_counter_node)
    builder.add_node("text_transformer_node", text_transformer_node)

    # Set entry point
    builder.set_entry_point("router")

    # Add conditional edges from router
    builder.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "calculator": "calculator_node",
            "string_reverser": "string_reverser_node",
            "word_counter": "word_counter_node",
            "text_transformer": "text_transformer_node"
        }
    )

    # All tools lead to END
    builder.add_edge("calculator_node", END)
    builder.add_edge("string_reverser_node", END)
    builder.add_edge("word_counter_node", END)
    builder.add_edge("text_transformer_node", END)

    return builder.compile()


def create_llm_tool_agent(mock_mode: bool = True):
    """
    Create an LLM agent that dynamically selects the appropriate tool.

    Graph: classify → should_use_tool? → [execute_tool → format_response | direct_response] → END

    Args:
        mock_mode: If True, use mock LLM responses
    """

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def string_reverser(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    @tool
    def word_counter(text: str) -> str:
        """Count words and characters."""
        words = len(text.split())
        chars = len(text)
        return f"Words: {words}, Characters: {chars}"

    @tool
    def text_transformer(text: str) -> str:
        """Transform text case."""
        return f"Upper: {text.upper()} | Lower: {text.lower()} | Title: {text.title()}"

    def classify_node(state: LLMAgentState) -> dict:
        """LLM classifies the question and selects tool."""
        question = state["question"]

        if mock_mode:
            # Mock classification logic
            q_lower = question.lower()
            if any(op in question for op in ['+', '-', '*', '/', 'calculate', 'compute']):
                tool_name = "calculator"
            elif 'reverse' in q_lower:
                tool_name = "string_reverser"
            elif 'count' in q_lower or 'words' in q_lower or 'characters' in q_lower:
                tool_name = "word_counter"
            elif any(word in q_lower for word in ['upper', 'lower', 'case', 'title', 'transform']):
                tool_name = "text_transformer"
            else:
                tool_name = "none"
        else:
            # Real LLM classification
            llm = ChatOpenAI(model="gpt-4")
            prompt = f"""Analyze this question and choose the best tool:

Question: {question}

Tools available:
- calculator: for math calculations
- string_reverser: to reverse text
- word_counter: to count words/characters
- text_transformer: to change text case
- none: if no tool needed

Respond with ONLY the tool name."""

            response = llm.invoke([HumanMessage(content=prompt)])
            tool_name = response.content.strip().lower()

        return {"tool_to_use": tool_name}

    def execute_tool_node(state: LLMAgentState) -> dict:
        """Execute the selected tool."""
        tool_name = state["tool_to_use"]
        question = state["question"]

        # Extract the relevant input from the question
        if tool_name == "calculator":
            # Extract math expression
            import re
            math_pattern = r'[\d\s\+\-\*\/\(\)\.]+'
            matches = re.findall(math_pattern, question)
            input_val = matches[0] if matches else question
            result = calculator.invoke({"expression": input_val})
        elif tool_name == "string_reverser":
            # Extract text to reverse
            words = question.split()
            input_val = words[-1] if words else question
            result = string_reverser.invoke({"text": input_val})
        elif tool_name == "word_counter":
            # Extract text after "in:"
            if "in:" in question.lower():
                input_val = question.split("in:", 1)[1].strip()
            else:
                input_val = question
            result = word_counter.invoke({"text": input_val})
        elif tool_name == "text_transformer":
            # Extract text to transform
            words = question.split()
            input_val = words[-1] if len(words) > 1 else question
            result = text_transformer.invoke({"text": input_val})
        else:
            result = "No tool selected"

        return {"tool_result": result}

    def format_response_node(state: LLMAgentState) -> dict:
        """Format the final response."""
        tool_result = state.get("tool_result", "")
        if tool_result:
            answer = f"Using {state['tool_to_use']}: {tool_result}"
        else:
            answer = "I can help with calculations, text reversal, word counting, or text transformation. Please ask a specific question!"
        return {"final_answer": answer}

    def direct_response_node(state: LLMAgentState) -> dict:
        """Provide direct response without tool."""
        return {"final_answer": "I can help with calculations, text reversal, word counting, or text transformation. Please ask a specific question!"}

    def should_use_tool(state: LLMAgentState) -> str:
        """Decide whether to use a tool or respond directly."""
        valid_tools = ["calculator", "string_reverser", "word_counter", "text_transformer"]
        if state["tool_to_use"] in valid_tools:
            return "use_tool"
        return "direct_response"

    # Build the graph
    builder = StateGraph(LLMAgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("format_response", format_response_node)
    builder.add_node("direct_response", direct_response_node)

    # Set entry point
    builder.set_entry_point("classify")

    # Conditional routing based on classification
    builder.add_conditional_edges(
        "classify",
        should_use_tool,
        {
            "use_tool": "execute_tool",
            "direct_response": "direct_response"
        }
    )

    # Tool execution leads to formatting
    builder.add_edge("execute_tool", "format_response")

    # Both paths lead to END
    builder.add_edge("format_response", END)
    builder.add_edge("direct_response", END)

    return builder.compile()


def create_multi_step_workflow(mock_mode: bool = True, max_iter: int = 3):
    """
    Create a multi-step reasoning workflow.

    Graph: plan → research → synthesize → should_continue? → [research (loop) | END]

    Args:
        mock_mode: If True, use mock LLM responses
        max_iter: Maximum iterations
    """

    def plan_node(state: MultiStepState) -> dict:
        """Break question into research steps."""
        question = state["question"]

        if mock_mode:
            plan = f"""Research Plan for: "{question}"

1. Define core concepts and terminology
2. Identify key applications and examples
3. Analyze benefits and limitations"""
        else:
            llm = ChatOpenAI(model="gpt-4")
            prompt = f"""Break this complex question into 3 specific research steps:

Question: {question}

Return ONLY a numbered list of research steps."""

            response = llm.invoke([HumanMessage(content=prompt)])
            plan = response.content

        return {"plan": plan, "iteration_count": 0}

    def research_node(state: MultiStepState) -> dict:
        """Research the next step."""
        plan = state["plan"]
        iteration = state["iteration_count"]

        if mock_mode:
            research_notes = [
                "Core concepts researched: Key definitions and foundational principles identified.",
                "Applications found: Multiple real-world use cases across different industries.",
                "Analysis complete: Strengths and weaknesses documented with evidence."
            ]
            note = research_notes[min(iteration, len(research_notes) - 1)]
        else:
            llm = ChatOpenAI(model="gpt-4")
            steps = [s.strip() for s in plan.split('\n') if s.strip() and s[0].isdigit()]

            if iteration < len(steps):
                step = steps[iteration]
                prompt = f"Research this step thoroughly: {step}"
                response = llm.invoke([HumanMessage(content=prompt)])
                note = response.content
            else:
                note = "Research step completed."

        return {
            "research_notes": [note],
            "iteration_count": iteration + 1
        }

    def synthesize_node(state: MultiStepState) -> dict:
        """Synthesize research into final answer."""
        question = state["question"]
        notes = state.get("research_notes", [])

        if mock_mode:
            synthesis = f"""Based on comprehensive research:

{question}

**Summary:** The research reveals multiple interconnected aspects with practical implications across various domains. The evidence suggests significant potential with some notable considerations for implementation.

**Key Findings:**
{chr(10).join(f'- {note}' for note in notes)}

**Conclusion:** The analysis demonstrates clear value propositions while acknowledging areas requiring careful attention."""
        else:
            llm = ChatOpenAI(model="gpt-4")
            notes_text = "\n".join(f"{i+1}. {note}" for i, note in enumerate(notes))
            prompt = f"""Synthesize these research notes into a comprehensive answer:

Question: {question}

Research Notes:
{notes_text}

Provide a well-structured, complete answer."""

            response = llm.invoke([HumanMessage(content=prompt)])
            synthesis = response.content

        return {"synthesis": synthesis}

    def should_continue(state: MultiStepState) -> str:
        """Decide whether to continue research."""
        iteration = state["iteration_count"]
        max_iterations = state["max_iterations"]

        # Continue if under max and haven't researched all steps
        if iteration < max_iterations and iteration < 3:
            return "research"
        return "finish"

    # Build the graph
    builder = StateGraph(MultiStepState)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)

    # Set entry point
    builder.set_entry_point("plan")

    # Linear flow with loop
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")

    # Conditional edge for loop
    builder.add_conditional_edges(
        "synthesize",
        should_continue,
        {
            "research": "research",
            "finish": END
        }
    )

    return builder.compile()


def create_multi_agent_graph(mock_mode: bool = True, max_revisions: int = 2):
    """
    Create a multi-agent collaboration graph (Writer + Critic).

    Graph: writer_draft → critic_review → should_revise? → [writer_revise → critic_review (loop) | END]

    Args:
        mock_mode: If True, use mock LLM responses
        max_revisions: Maximum number of revisions
    """

    def writer_draft_node(state: MultiAgentState) -> dict:
        """Writer agent creates initial draft."""
        topic = state["topic"]

        if mock_mode:
            draft = f"""# {topic}

LangGraph is a powerful framework for building stateful, multi-agent applications. It provides several key advantages:

**Flexible Workflows:** Unlike linear chains, LangGraph supports cycles and conditional branching, enabling complex decision-making patterns.

**State Management:** Built-in state handling makes it easy to maintain context across multiple steps, with typed state definitions for clarity.

**Production Ready:** The framework includes features like checkpointing, streaming, and error handling that are essential for real-world applications."""
        else:
            llm = ChatOpenAI(model="gpt-4", temperature=0.7)
            prompt = f"""You are a technical writer. Write a clear, concise explanation about:

Topic: {topic}

Write 2-3 paragraphs. Be accurate and engaging."""

            draft = llm.invoke([HumanMessage(content=prompt)]).content

        return {"draft": draft, "revision_count": 0}

    def critic_review_node(state: MultiAgentState) -> dict:
        """Critic agent reviews the draft."""
        draft = state["draft"]

        if mock_mode:
            # Mock: Approve on second iteration, give feedback on first
            revision_count = state.get("revision_count", 0)
            if revision_count > 0:
                feedback = "APPROVED: The revised draft is excellent. Clear structure, accurate information, and good examples."
                approved = True
            else:
                feedback = "Good start, but please add specific examples of how to use LangGraph and mention its integration with LangChain."
                approved = False
        else:
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)
            prompt = f"""You are a critic reviewing technical content.

Draft: {draft}

Provide either:
1. "APPROVED" if the draft is excellent
2. Specific, actionable feedback for improvement

Keep feedback concise and constructive."""

            feedback = llm.invoke([HumanMessage(content=prompt)]).content
            approved = "APPROVED" in feedback.upper()

        return {
            "critic_feedback": [feedback],
            "approved": approved
        }

    def writer_revise_node(state: MultiAgentState) -> dict:
        """Writer agent revises based on feedback."""
        draft = state["draft"]
        feedback_list = state.get("critic_feedback", [])
        latest_feedback = feedback_list[-1] if feedback_list else ""

        if mock_mode:
            # Mock revision
            revised = f"""# {state['topic']}

LangGraph is a powerful framework for building stateful, multi-agent applications. It provides several key advantages:

**Flexible Workflows:** Unlike linear chains, LangGraph supports cycles and conditional branching, enabling complex decision-making patterns.

**State Management:** Built-in state handling makes it easy to maintain context across multiple steps, with typed state definitions for clarity.

**Practical Example:** To build a simple graph, you define a state TypedDict, create nodes as functions, and connect them with edges. For instance, you can create a ReAct agent that reasons, acts with tools, and observes results in a loop.

**Production Ready:** The framework integrates seamlessly with LangChain's ecosystem of LLMs and tools, while adding features like checkpointing, streaming, and error handling essential for real-world applications."""
        else:
            llm = ChatOpenAI(model="gpt-4", temperature=0.7)
            prompt = f"""Revise your draft based on this feedback:

Original Draft: {draft}

Feedback: {latest_feedback}

Write an improved version addressing the feedback."""

            revised = llm.invoke([HumanMessage(content=prompt)]).content

        return {
            "draft": revised,
            "revision_count": state["revision_count"] + 1,
            "approved": False  # Reset approval status
        }

    def should_revise(state: MultiAgentState) -> str:
        """Decide whether to revise or finish."""
        if state.get("approved", False):
            return "finish"
        elif state.get("revision_count", 0) >= state.get("max_revisions", max_revisions):
            return "finish"
        return "revise"

    # Build the graph
    builder = StateGraph(MultiAgentState)
    builder.add_node("writer_draft", writer_draft_node)
    builder.add_node("critic_review", critic_review_node)
    builder.add_node("writer_revise", writer_revise_node)

    # Set entry point
    builder.set_entry_point("writer_draft")

    # Linear flow to first review
    builder.add_edge("writer_draft", "critic_review")

    # Conditional edge for revision loop
    builder.add_conditional_edges(
        "critic_review",
        should_revise,
        {
            "revise": "writer_revise",
            "finish": END
        }
    )

    # Revised draft goes back to critic
    builder.add_edge("writer_revise", "critic_review")

    return builder.compile()
