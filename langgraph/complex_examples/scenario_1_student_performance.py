"""
Scenario 1: Student Performance Query with ReAct Pattern

This script demonstrates the ReAct (Reasoning + Acting) pattern using LangGraph,
with automatic retry logic and error recovery.

Key Concepts Demonstrated:
- ReAct Pattern: Iterative reasoning → acting → observing cycle
- Retry Logic: Graph can loop back to reason node on errors
- State Management: Using Annotated[list, operator.add] for message accumulation
- Tool Integration: LangChain tools with @tool decorator
- Conditional Routing: Dynamic edge routing based on state
- Error Recovery: LLM sees previous errors and adjusts strategy

Graph Structure:
    START → reason → [extract_id → call_tool → observe → check_result]
                                                              ├─→ [Error + retries left] reason (LOOP BACK)
                                                              └─→ [Success OR max retries] generate_answer → END

Usage:
    # Requires OPENAI_API_KEY in .env file
    uv run python complex_examples/scenario_1_student_performance.py

Author: Claude Code
Date: 2026-02-05
"""

# ============================================================================
# Imports
# ============================================================================

import os
import json
import sys
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import argparse

# ============================================================================
# Configuration
# ============================================================================

# Student database constants
KNOWN_STUDENTS = """
Available students:
- S12345: John Doe (High performer)
- S23456: Jane Smith (Good performer)
- S34567: Bob Johnson (Needs support)
"""

# ============================================================================
# State Definition
# ============================================================================

class ReActStudentState(TypedDict):
    """State for ReAct-based student performance query workflow.

    The ReAct pattern maintains conversation history to enable the LLM
    to learn from previous attempts and adjust its strategy on retries.
    """
    user_question: str                              # Original natural language question
    messages: Annotated[list, operator.add]         # Accumulated conversation history (reasoning trace)
    student_id: str                                 # Extracted/identified student ID
    performance_data: str                           # JSON string with student info from tool
    iteration_count: int                            # Current iteration number
    max_iterations: int                             # Maximum allowed iterations (prevents infinite loops)
    final_answer: str                               # Natural language response to user

# ============================================================================
# Tools
# ============================================================================

@tool
def get_student_performance(student_id: str) -> str:
    """Retrieve student performance data by student ID.

    This tool simulates a database lookup for student academic records.
    In a real system, this would query a database or API.

    Args:
        student_id: Unique student identifier (format: SXXXXX)

    Returns:
        JSON string containing student performance information including
        grades, GPA, attendance, and behavioral notes.

    Example:
        >>> result = get_student_performance.invoke({"student_id": "S12345"})
        >>> data = json.loads(result)
        >>> print(data["name"])
        "John Doe"
    """
    # Mock database of student records
    STUDENT_DATABASE = {
        "S12345": {
            "student_id": "S12345",
            "name": "John Doe",
            "email": "john.doe@school.edu",
            "grades": {
                "Math": {"grade": "A", "score": 92},
                "Science": {"grade": "B+", "score": 88},
                "English": {"grade": "A-", "score": 90}
            },
            "attendance_percentage": 94.5,
            "behavioral_notes": "Excellent participation in class. Shows strong leadership skills.",
            "gpa": 3.67,
            "class_rank": "12/150",
            "semester": "Fall 2024"
        },
        "S23456": {
            "student_id": "S23456",
            "name": "Jane Smith",
            "email": "jane.smith@school.edu",
            "grades": {
                "Math": {"grade": "B", "score": 85},
                "Science": {"grade": "A", "score": 95},
                "English": {"grade": "B+", "score": 88}
            },
            "attendance_percentage": 91.2,
            "behavioral_notes": "Demonstrates strong analytical skills. Could improve time management.",
            "gpa": 3.45,
            "class_rank": "28/150",
            "semester": "Fall 2024"
        },
        "S34567": {
            "student_id": "S34567",
            "name": "Bob Johnson",
            "email": "bob.johnson@school.edu",
            "grades": {
                "Math": {"grade": "C+", "score": 78},
                "Science": {"grade": "C", "score": 75},
                "English": {"grade": "B-", "score": 82}
            },
            "attendance_percentage": 87.8,
            "behavioral_notes": "Has potential but needs to focus more on homework completion.",
            "gpa": 2.89,
            "class_rank": "89/150",
            "semester": "Fall 2024"
        }
    }

    # Lookup student
    if student_id in STUDENT_DATABASE:
        return json.dumps(STUDENT_DATABASE[student_id], indent=2)
    else:
        # Return error JSON for not found
        return json.dumps({
            "error": f"Student {student_id} not found in database",
            "available_ids": list(STUDENT_DATABASE.keys()),
            "suggestion": "Please check the student ID or try using a student name from the available list"
        })

# ============================================================================
# Node Functions
# ============================================================================

def reason_node(state: ReActStudentState) -> dict:
    """LLM reasons about what action to take next.

    This is the core of the ReAct pattern. The LLM:
    1. Reviews the conversation history (including previous errors)
    2. Decides what to do next based on current state
    3. Appends its reasoning to the message history

    On retries, the LLM can see what went wrong and adjust strategy.
    """
    print("  → Node: reason")

    question = state["user_question"]
    messages = state.get("messages", [])
    student_id = state.get("student_id", "")
    performance_data = state.get("performance_data", "")
    iteration = state.get("iteration_count", 0)

    # Build context for LLM
    context = f"""You are helping analyze a student performance query.

Question: {question}

{KNOWN_STUDENTS}

Current state:
- Student ID extracted: {student_id if student_id else "Not yet"}
- Performance data retrieved: {"Yes" if performance_data else "No"}
- Iteration: {iteration + 1}

Previous attempts and results are in the conversation history above.

Based on the current state, what should we do next?
- If we don't have a student ID yet, we need to identify which student
- If we have an ID but no data (or error), we may need to retry or try a different approach
- If we have data, we're ready to generate the final answer

Respond with your reasoning about what to do next."""

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Include conversation history for context
        full_messages = messages + [HumanMessage(content=context)]
        response = llm.invoke(full_messages)

        reasoning = response.content
        print(f"    Reasoning: {reasoning[:100]}...")

        # Append reasoning to messages - this builds the reasoning trace
        return {"messages": [AIMessage(content=f"[Iteration {iteration + 1}] Reasoning: {reasoning}")]}

    except Exception as e:
        print(f"    ❌ Error in reason node: {str(e)}")
        return {"messages": [AIMessage(content=f"Error during reasoning: {str(e)}")]}


def extract_id_node(state: ReActStudentState) -> dict:
    """Extract student ID from the question using LLM.

    The LLM can identify students by:
    - Direct ID mention (e.g., "S12345")
    - Student name (e.g., "John Doe")
    - Partial name (e.g., "John")
    """
    print("  → Node: extract_id")

    question = state["user_question"]

    prompt = f"""Extract the student identifier from this question.

Question: {question}

{KNOWN_STUDENTS}

Respond with ONLY the student ID in format SXXXXX (e.g., S12345).
If you can identify the student by name, return their ID.
If no clear match, return S12345 as default."""

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])

        student_id = response.content.strip().upper()
        print(f"    Extracted: {student_id}")

        return {
            "student_id": student_id,
            "messages": [AIMessage(content=f"Extracted student ID: {student_id}")]
        }

    except Exception as e:
        print(f"    ❌ Error extracting ID: {str(e)}")
        return {
            "student_id": "S12345",  # Default fallback
            "messages": [AIMessage(content=f"Error extracting ID: {str(e)}, using default S12345")]
        }


def call_tool_node(state: ReActStudentState) -> dict:
    """Invoke the get_student_performance tool.

    This demonstrates tool invocation in LangGraph.
    The tool is called with .invoke() and passed a parameter dict.
    """
    print("  → Node: call_tool")

    student_id = state["student_id"]

    try:
        # Tool invocation pattern: pass params as dict
        result = get_student_performance.invoke({"student_id": student_id})

        # Check if tool returned error
        data = json.loads(result)
        if "error" in data:
            print(f"    ⚠️  Tool returned error: {data['error']}")
        else:
            print(f"    ✓ Retrieved data for {data.get('name', 'Unknown')}")

        return {
            "performance_data": result,
            "messages": [AIMessage(content=f"Called tool with student_id={student_id}, got result")]
        }

    except Exception as e:
        print(f"    ❌ Tool invocation error: {str(e)}")
        error_result = json.dumps({"error": str(e)})
        return {
            "performance_data": error_result,
            "messages": [AIMessage(content=f"Tool error: {str(e)}")]
        }


def observe_node(state: ReActStudentState) -> dict:
    """Observe the tool result and update iteration count.

    This node:
    1. Checks if the tool call was successful
    2. Increments iteration counter
    3. Adds observation to message history

    The router function will use this information to decide whether to retry.
    """
    print("  → Node: observe")

    performance_data = state["performance_data"]
    iteration = state.get("iteration_count", 0)

    # Parse result to check for errors
    try:
        data = json.loads(performance_data)

        if "error" in data:
            observation = f"Tool call failed: {data['error']}"
            print(f"    Observation: Error detected - {data.get('suggestion', 'No suggestion')}")
        else:
            observation = f"Successfully retrieved data for {data.get('name', 'student')}"
            print(f"    Observation: Success - got data for {data.get('name', 'student')}")

        return {
            "iteration_count": iteration + 1,
            "messages": [AIMessage(content=f"Observation: {observation}")]
        }

    except Exception as e:
        print(f"    ⚠️  Error parsing result: {str(e)}")
        return {
            "iteration_count": iteration + 1,
            "messages": [AIMessage(content=f"Observation: Error parsing result - {str(e)}")]
        }


def generate_answer_node(state: ReActStudentState) -> dict:
    """Generate final natural language answer using LLM.

    This node synthesizes the performance data into a clear,
    helpful response to the original question.
    """
    print("  → Node: generate_answer")

    performance_data = state["performance_data"]
    question = state["user_question"]
    iteration = state.get("iteration_count", 0)

    # Parse data
    data = json.loads(performance_data)

    # Handle error case
    if "error" in data:
        answer = f"""I apologize, but I couldn't find the requested student information.

Error: {data['error']}

{data.get('suggestion', '')}

Available students: {', '.join(data.get('available_ids', []))}

After {iteration} attempt(s), I was unable to retrieve valid student data. Please verify the student ID or name and try again."""

        print("    Generated error response")
        return {"final_answer": answer}

    # Generate answer with LLM
    prompt = f"""Based on this student performance data, answer the following question clearly and concisely.

Question: {question}

Student Data:
{json.dumps(data, indent=2)}

Provide a helpful, encouraging response that directly answers the question.
Include specific details about grades, GPA, attendance, and any relevant notes.
Keep it to 2-3 short paragraphs."""

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        response = llm.invoke([HumanMessage(content=prompt)])

        answer = response.content
        print("    ✓ Generated final answer with LLM")

        return {"final_answer": answer}

    except Exception as e:
        print(f"    ⚠️  LLM error: {str(e)}, using basic response")

        # Fallback: basic formatted response
        name = data.get("name", "Unknown")
        gpa = data.get("gpa", "N/A")
        grades = data.get("grades", {})

        grade_text = ", ".join([
            f"{subj}: {info['grade']}"
            for subj, info in grades.items()
        ])

        answer = f"{name}'s performance: GPA {gpa}, Grades: {grade_text}"
        return {"final_answer": answer}

# ============================================================================
# Router Functions
# ============================================================================

def route_after_reason(state: ReActStudentState) -> str:
    """Route based on what we have so far.

    This demonstrates conditional routing in LangGraph.
    The router function examines state and returns a string key
    that maps to the next node to execute.
    """
    student_id = state.get("student_id", "")
    performance_data = state.get("performance_data", "")

    if not student_id:
        # Need to extract student ID first
        return "extract_id"
    elif not performance_data:
        # Have ID, need to call tool
        return "call_tool"
    else:
        # Have data, ready to generate answer
        # (This path handles successful first attempt)
        return "generate_answer"


def route_after_observe(state: ReActStudentState) -> str:
    """Route based on observation result.

    This implements the retry logic:
    - If error and retries available: loop back to reason node
    - Otherwise: proceed to generate answer
    """
    performance_data = state["performance_data"]
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    # Check if tool call had an error
    try:
        data = json.loads(performance_data)
        has_error = "error" in data
    except:
        has_error = True

    # Decide whether to retry
    if has_error and iteration < max_iterations:
        print(f"    Routing: Error detected, retrying (iteration {iteration}/{max_iterations})")
        return "reason"  # Loop back to reason node for retry
    else:
        if has_error:
            print(f"    Routing: Max iterations reached, generating answer with available info")
        else:
            print(f"    Routing: Success, generating answer")
        return "generate_answer"

# ============================================================================
# Graph Construction
# ============================================================================

def create_react_student_graph(max_iterations: int = 3):
    """Create and compile the ReAct-based student query graph.

    Graph demonstrates:
    - Conditional routing based on state
    - Loops/cycles for retry logic
    - Multiple decision points
    - Message accumulation for reasoning trace

    Args:
        max_iterations: Maximum number of retry attempts (default: 3)

    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize graph with state type
    builder = StateGraph(ReActStudentState)

    # Add nodes
    builder.add_node("reason", reason_node)
    builder.add_node("extract_id", extract_id_node)
    builder.add_node("call_tool", call_tool_node)
    builder.add_node("observe", observe_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Set entry point
    builder.set_entry_point("reason")

    # Add conditional routing after reason node
    # Router function returns string key that maps to next node
    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "extract_id": "extract_id",
            "call_tool": "call_tool",
            "generate_answer": "generate_answer"
        }
    )

    # Linear flow: extract_id → call_tool → observe
    builder.add_edge("extract_id", "call_tool")
    builder.add_edge("call_tool", "observe")

    # Conditional routing after observe - this enables retry loop
    builder.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "reason": "reason",  # Loop back for retry
            "generate_answer": "generate_answer"  # Success path
        }
    )

    # Terminal node
    builder.add_edge("generate_answer", END)

    return builder.compile()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for scenario 1.

    Demonstrates:
    - Environment validation
    - Graph creation and execution
    - Reasoning trace display
    - Iteration counting
    """
    # Load environment variables
    load_dotenv()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("\n" + "="*70)
        print("❌ Error: Valid OPENAI_API_KEY required")
        print("="*70)
        print("\nThis script requires a valid OpenAI API key in your .env file.")
        print("Please add: OPENAI_API_KEY=sk-...")
        print("\nExiting.\n")
        sys.exit(1)

    # Print banner
    print("\n" + "="*70)
    print(" "*15 + "SCENARIO 1: Student Performance Query")
    print(" "*20 + "(ReAct Pattern with Retry)")
    print("="*70)

    print("\nThis demonstrates the ReAct (Reasoning + Acting) pattern:")
    print("  • LLM reasons about what to do next")
    print("  • Executes actions (tool calls)")
    print("  • Observes results")
    print("  • Retries with adjusted strategy on errors")
    print("  • Maximum 3 iterations to prevent infinite loops\n")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query student performance using ReAct pattern")
    parser.add_argument("question", nargs="*", help="Question to ask (e.g., 'What are John Doe grades?')")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enter question interactively")
    args = parser.parse_args()

    # Determine question source
    if args.interactive or not args.question:
        # Interactive mode: prompt user for question
        print("💬 Interactive Mode")
        print(f"\n{KNOWN_STUDENTS}")
        print("-" * 70)
        question = input("Enter your question about student performance: ").strip()
        if not question:
            print("❌ No question provided. Exiting.\n")
            sys.exit(1)
        print()
    else:
        # Command-line mode: use provided arguments
        question = " ".join(args.question)

    # Create graph
    print("Building graph...")
    graph = create_react_student_graph(max_iterations=3)
    print("✓ Graph compiled successfully\n")

    print(f"Question: \"{question}\"")
    print("\nExecuting graph with ReAct pattern...\n")
    print("-" * 70)

    try:
        # Invoke graph - this runs the ReAct loop
        result = graph.invoke({
            "user_question": question,
            "messages": [],
            "student_id": "",
            "performance_data": "",
            "iteration_count": 0,
            "max_iterations": 3,
            "final_answer": ""
        })

        print("-" * 70)

        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        print(f"\n📊 Iterations completed: {result['iteration_count']}")
        print(f"🎯 Student ID: {result['student_id']}")

        # Show reasoning trace
        print(f"\n🧠 Reasoning Trace ({len(result['messages'])} steps):")
        print("-" * 70)
        for i, msg in enumerate(result['messages'], 1):
            content = msg.content
            # Truncate long messages
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"{i}. {content}")

        # Show final answer
        print("\n" + "-" * 70)
        print("📝 Final Answer:")
        print("-" * 70)
        print(result['final_answer'])
        print("-" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user (Ctrl+C)")
        print("Exiting gracefully...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
