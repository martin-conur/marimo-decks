"""
Scenario 2: Student Email Workflow with Human-in-the-Loop

This script demonstrates human-in-the-loop (HITL) pattern using LangGraph,
with conditional routing and multiple decision points.

Key Concepts Demonstrated:
- Human-in-the-Loop: Pause execution for user decisions
- Conditional Routing: Branch to different nodes based on decisions
- Multiple Terminal Nodes: Different endpoints for different outcomes
- Terminal Input: Using input() to capture user decisions
- State-Based Branching: Router functions read state to determine flow

Graph Structure:
    START → extract_student_id → retrieve_performance → human_checkpoint_1
                                                              ├─→ [No] cancel_workflow → END
                                                              └─→ [Yes] draft_email → human_checkpoint_2
                                                                                            ├─→ [No] discard_email → END
                                                                                            └─→ [Yes] send_email → END

Usage:
    # Requires OPENAI_API_KEY in .env file
    uv run python complex_examples/scenario_2_student_email.py

    # You will be prompted twice:
    # 1. After seeing performance data: "Send email?"
    # 2. After seeing draft: "Approve and send?"

Author: Claude Code
Date: 2026-02-05
"""

# ============================================================================
# Imports
# ============================================================================

import os
import json
import sys
import time
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import argparse

# ============================================================================
# Configuration
# ============================================================================

KNOWN_STUDENTS = """
Available students:
- S12345: John Doe
- S23456: Jane Smith
- S34567: Bob Johnson
"""

# ============================================================================
# State Definition
# ============================================================================

class StudentEmailState(TypedDict):
    """State for student email workflow with human-in-the-loop.

    This demonstrates state management for branching workflows with multiple
    decision points and conditional routing based on user input.
    """
    user_question: str          # Original request
    student_id: str             # Extracted student ID
    performance_data: str       # JSON string with student info
    should_send_email: bool     # HITL checkpoint 1: send email?
    email_draft: str            # Generated email content
    email_approved: bool        # HITL checkpoint 2: approve draft?
    final_status: str           # Workflow outcome message

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
        return json.dumps({
            "error": f"Student {student_id} not found in database",
            "available_ids": list(STUDENT_DATABASE.keys())
        })

# ============================================================================
# Node Functions
# ============================================================================

def extract_student_id_node(state: StudentEmailState) -> dict:
    """Extract student ID from request using LLM.

    The LLM can identify students by ID or name.
    This demonstrates LLM-based extraction in a workflow.
    """
    print("  → Node: extract_student_id")

    question = state["user_question"]

    prompt = f"""Extract the student identifier from this request.

Request: {question}

{KNOWN_STUDENTS}

Respond with ONLY the student ID in format SXXXXX (e.g., S12345).
If you can identify the student by name, return their ID.
If no clear match, return S12345 as default."""

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])

        student_id = response.content.strip().upper()
        print(f"    Extracted: {student_id}")

        return {"student_id": student_id}

    except Exception as e:
        print(f"    ❌ Error extracting ID: {str(e)}")
        return {"student_id": "S12345"}  # Default fallback


def retrieve_performance_node(state: StudentEmailState) -> dict:
    """Retrieve student performance data using the tool.

    This demonstrates tool invocation in a workflow node.
    """
    print("  → Node: retrieve_performance")

    student_id = state["student_id"]

    try:
        # Tool invocation pattern
        result = get_student_performance.invoke({"student_id": student_id})

        # Validate result
        data = json.loads(result)
        if "error" in data:
            print(f"    ❌ {data['error']}")
        else:
            print(f"    ✓ Retrieved data for {data.get('name', 'Unknown')}")

        return {"performance_data": result}

    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {"performance_data": json.dumps({"error": str(e)})}


def human_checkpoint_1_node(state: StudentEmailState) -> dict:
    """HITL Checkpoint 1: Ask if user wants to proceed with email.

    This demonstrates:
    - Pausing graph execution for human input
    - Displaying relevant context before decision
    - Capturing user choice in state using terminal input()

    The graph will branch based on the boolean returned here.
    """
    print("\n" + "="*70)
    print(" "*20 + "🔍 CHECKPOINT 1: Review Performance Data")
    print("="*70)

    # Parse and display key information
    data = json.loads(state["performance_data"])

    if "error" in data:
        print(f"\n❌ Error: {data['error']}")
        print("Cannot proceed with email workflow.")
        print("\n" + "="*70 + "\n")
        return {"should_send_email": False}

    # Display summary for decision-making
    print(f"\n📊 Student Information:")
    print(f"  Name: {data.get('name', 'Unknown')}")
    print(f"  Email: {data.get('email', 'unknown@school.edu')}")
    print(f"  GPA: {data.get('gpa', 'N/A')}")
    print(f"  Class Rank: {data.get('class_rank', 'N/A')}")
    print(f"  Attendance: {data.get('attendance_percentage', 'N/A')}%")

    print(f"\n📚 Grades:")
    for subject, info in data.get("grades", {}).items():
        print(f"  • {subject}: {info['grade']} ({info['score']}%)")

    print(f"\n📝 Behavioral Notes:")
    print(f"  {data.get('behavioral_notes', 'None')}")

    # Ask for decision - this PAUSES execution until user responds
    print("\n" + "-"*70)
    response = input("Would you like to draft an email to this student? (yes/no): ").strip().lower()

    should_send = response in ['yes', 'y']

    if should_send:
        print("✓ Proceeding to draft email...\n")
    else:
        print("✗ Workflow cancelled.\n")

    print("="*70 + "\n")

    return {"should_send_email": should_send}


def draft_email_node(state: StudentEmailState) -> dict:
    """Generate email draft based on student performance using LLM.

    This demonstrates:
    - Content generation from structured data
    - LLM-based personalization
    - Adapting tone based on performance level
    """
    print("  → Node: draft_email")

    data = json.loads(state["performance_data"])

    prompt = f"""Write a professional email to a student about their academic performance.

Student Data:
{json.dumps(data, indent=2)}

The email should:
- Include an appropriate subject line
- Be encouraging and supportive in tone
- Mention specific achievements or areas for improvement
- Be concise (2-3 paragraphs)
- End with an offer to discuss further
- Sign off as "Academic Advisor"

Write the complete email including the "Subject:" line."""

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        response = llm.invoke([HumanMessage(content=prompt)])

        email = response.content
        print("    ✓ Generated personalized email")

        return {"email_draft": email}

    except Exception as e:
        print(f"    ❌ LLM error: {str(e)}")

        # Fallback: basic template
        name = data.get("name", "Student")
        gpa = data.get("gpa", "N/A")

        email = f"""Subject: Academic Update

Dear {name.split()[0]},

I wanted to reach out regarding your academic progress this semester. Your current GPA is {gpa}.

Please feel free to reach out if you'd like to discuss your performance.

Best regards,
Academic Advisor"""

        print("    ⚠️  Using fallback template")
        return {"email_draft": email}


def human_checkpoint_2_node(state: StudentEmailState) -> dict:
    """HITL Checkpoint 2: Review and approve email draft.

    This demonstrates:
    - Presenting generated content for review
    - Second decision point in workflow
    - Final approval before action

    This is the second pause point in the workflow.
    """
    print("\n" + "="*70)
    print(" "*22 + "📧 CHECKPOINT 2: Review Email Draft")
    print("="*70)

    # Display draft for review
    print("\n" + "-"*70)
    print(state["email_draft"])
    print("-"*70)

    # Ask for approval - execution pauses here
    print("\n" + "-"*70)
    response = input("Send this email? (yes/no): ").strip().lower()

    approved = response in ['yes', 'y']

    if approved:
        print("✓ Email approved. Sending...\n")
    else:
        print("✗ Email discarded.\n")

    print("="*70 + "\n")

    return {"email_approved": approved}


def send_email_node(state: StudentEmailState) -> dict:
    """Simulate sending the email.

    This demonstrates a terminal/success node.
    In a real system, this would call an email API.
    """
    print("  → Node: send_email")

    data = json.loads(state["performance_data"])
    recipient = data.get("email", "unknown@school.edu")
    name = data.get("name", "student")

    print(f"    📤 Sending email to {recipient}...")
    time.sleep(1)  # Simulate sending delay
    print("    ✓ Email sent successfully!")

    return {"final_status": f"✓ Email sent to {name} ({recipient})"}


def cancel_workflow_node(state: StudentEmailState) -> dict:
    """Handle workflow cancellation at checkpoint 1.

    Terminal node for the "No" path at first checkpoint.
    """
    print("  → Node: cancel_workflow")
    print("    Workflow cancelled by user at checkpoint 1")

    return {"final_status": "✗ Workflow cancelled - no email sent"}


def discard_email_node(state: StudentEmailState) -> dict:
    """Handle email draft rejection at checkpoint 2.

    Terminal node for the "No" path at second checkpoint.
    """
    print("  → Node: discard_email")
    print("    Email draft discarded by user at checkpoint 2")

    return {"final_status": "🗑️  Email draft discarded - no email sent"}

# ============================================================================
# Router Functions
# ============================================================================

def decide_after_checkpoint_1(state: StudentEmailState) -> str:
    """Router function for checkpoint 1 decision.

    This demonstrates conditional routing in LangGraph.
    The router function reads state and returns a string key
    that determines which node to execute next.

    Returns:
        "draft_email" if user wants to proceed
        "cancel_workflow" if user cancelled
    """
    if state.get("should_send_email", False):
        return "draft_email"
    else:
        return "cancel_workflow"


def decide_after_checkpoint_2(state: StudentEmailState) -> str:
    """Router function for checkpoint 2 decision.

    Returns:
        "send_email" if user approved the draft
        "discard_email" if user rejected the draft
    """
    if state.get("email_approved", False):
        return "send_email"
    else:
        return "discard_email"

# ============================================================================
# Graph Construction
# ============================================================================

def create_student_email_graph():
    """Create and compile the student email workflow graph with HITL.

    This graph demonstrates:
    - Human-in-the-loop pattern with multiple checkpoints
    - Conditional routing to multiple terminal nodes
    - Branching workflow with 3 possible outcomes

    Graph structure:
        START → extract → retrieve → checkpoint_1
                                          ├─→ [No] cancel → END
                                          └─→ [Yes] draft → checkpoint_2
                                                                ├─→ [No] discard → END
                                                                └─→ [Yes] send → END

    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize graph with state type
    builder = StateGraph(StudentEmailState)

    # Add all nodes
    builder.add_node("extract_student_id", extract_student_id_node)
    builder.add_node("retrieve_performance", retrieve_performance_node)
    builder.add_node("human_checkpoint_1", human_checkpoint_1_node)
    builder.add_node("draft_email", draft_email_node)
    builder.add_node("human_checkpoint_2", human_checkpoint_2_node)
    builder.add_node("send_email", send_email_node)
    builder.add_node("cancel_workflow", cancel_workflow_node)
    builder.add_node("discard_email", discard_email_node)

    # Set entry point
    builder.set_entry_point("extract_student_id")

    # Linear flow up to first checkpoint
    builder.add_edge("extract_student_id", "retrieve_performance")
    builder.add_edge("retrieve_performance", "human_checkpoint_1")

    # Conditional routing after checkpoint 1
    # The router function returns a string that maps to the next node
    builder.add_conditional_edges(
        "human_checkpoint_1",
        decide_after_checkpoint_1,
        {
            "draft_email": "draft_email",        # User said yes
            "cancel_workflow": "cancel_workflow"  # User said no
        }
    )

    # Draft leads to checkpoint 2
    builder.add_edge("draft_email", "human_checkpoint_2")

    # Conditional routing after checkpoint 2
    builder.add_conditional_edges(
        "human_checkpoint_2",
        decide_after_checkpoint_2,
        {
            "send_email": "send_email",      # User approved
            "discard_email": "discard_email"  # User rejected
        }
    )

    # All terminal nodes lead to END
    builder.add_edge("send_email", END)
    builder.add_edge("cancel_workflow", END)
    builder.add_edge("discard_email", END)

    return builder.compile()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for scenario 2.

    Demonstrates:
    - Environment validation
    - Interactive workflow with user decisions
    - Multiple outcome paths
    - Decision trail tracking
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
    print(" "*12 + "SCENARIO 2: Student Email Workflow (with HITL)")
    print("="*70)

    print("\nThis workflow includes TWO human-in-the-loop checkpoints:")
    print("  1. After retrieving performance data: Should we draft an email?")
    print("  2. After drafting email: Should we send it?")
    print("\nYou'll be prompted to make decisions at each checkpoint.")
    print("The workflow branches based on your responses.\n")

    print("Possible outcomes:")
    print("  • Cancel at checkpoint 1 → No email drafted")
    print("  • Approve → Draft → Cancel at checkpoint 2 → Email discarded")
    print("  • Approve → Draft → Approve → Email sent ✓\n")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Send email to student with HITL workflow")
    parser.add_argument("request", nargs="*", help="Email request (e.g., 'Send email to Jane Smith')")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enter request interactively")
    args = parser.parse_args()

    # Determine request source
    if args.interactive or not args.request:
        # Interactive mode: prompt user for request
        print("💬 Interactive Mode")
        print(f"\n{KNOWN_STUDENTS}")
        print("-" * 70)
        request = input("Enter your email request: ").strip()
        if not request:
            print("❌ No request provided. Exiting.\n")
            sys.exit(1)
        print()
    else:
        # Command-line mode: use provided arguments
        request = " ".join(args.request)

    # Create graph
    print("Building graph...")
    graph = create_student_email_graph()
    print("✓ Graph compiled successfully\n")

    print(f"Request: \"{request}\"")
    print("\nStarting workflow...\n")
    print("-" * 70)

    try:
        # Invoke graph - will pause at HITL checkpoints
        result = graph.invoke({
            "user_question": request,
            "student_id": "",
            "performance_data": "",
            "should_send_email": False,
            "email_draft": "",
            "email_approved": False,
            "final_status": ""
        })

        print("-" * 70)

        # Display final outcome
        print("\n" + "="*70)
        print(" "*25 + "WORKFLOW COMPLETE")
        print("="*70)

        print(f"\n📍 Final Status: {result.get('final_status', 'Unknown')}")

        # Show decision trail
        print(f"\n🛤️  Decision Trail:")
        print(f"  Checkpoint 1 (Send email?): {'✓ Yes' if result.get('should_send_email') else '✗ No'}")

        if result.get('should_send_email'):
            print(f"  Checkpoint 2 (Approve draft?): {'✓ Yes' if result.get('email_approved') else '✗ No'}")
        else:
            print(f"  Checkpoint 2: (Not reached)")

        print("\n" + "="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  Workflow interrupted by user (Ctrl+C)")
        print("="*70)
        print("No email was sent.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during workflow: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
