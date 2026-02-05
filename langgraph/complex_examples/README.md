# Complex LangGraph Examples

This folder contains standalone educational scripts demonstrating advanced LangGraph patterns with real-world scenarios.

## Prerequisites

- Valid `OPENAI_API_KEY` in `.env` file (parent directory)
- Python 3.10+
- Dependencies installed via `uv sync`

## Scripts Overview

### Scenario 1: Student Performance Query (ReAct Pattern)

Demonstrates the **ReAct (Reasoning + Acting)** pattern with automatic retry logic and error recovery.

**Key Concepts:**
- Iterative reasoning loop
- Automatic retry on errors (max 3 iterations)
- Message accumulation for reasoning trace
- Conditional routing based on state
- Tool integration with error handling

**Graph Structure:**
```
START → reason → [extract_id → call_tool → observe → check_result]
                                                          ├─→ [Error + retries] reason (LOOP)
                                                          └─→ [Success] generate_answer → END
```

**Usage:**

```bash
# Default example question
uv run python complex_examples/scenario_1_student_performance.py

# Custom question via command line
uv run python complex_examples/scenario_1_student_performance.py "What is Bob Johnson's GPA?"

# Interactive mode (prompts for question)
uv run python complex_examples/scenario_1_student_performance.py --interactive
uv run python complex_examples/scenario_1_student_performance.py -i
```

**Example Questions:**
- "What are John Doe's grades?"
- "How is S23456 performing in Math?"
- "Tell me about Jane Smith's attendance"
- "What's Bob's class rank?"

---

### Scenario 2: Student Email Workflow (Human-in-the-Loop)

Demonstrates **Human-in-the-Loop (HITL)** pattern with conditional routing and multiple decision checkpoints.

**Key Concepts:**
- Human-in-the-loop with terminal `input()`
- Conditional routing to multiple terminal nodes
- Branching workflow with 3 possible outcomes
- LLM-generated personalized content
- State-based decision tracking

**Graph Structure:**
```
START → extract → retrieve → checkpoint_1
                                  ├─→ [No] cancel → END
                                  └─→ [Yes] draft → checkpoint_2
                                                        ├─→ [No] discard → END
                                                        └─→ [Yes] send → END
```

**Usage:**

```bash
# Default example request
uv run python complex_examples/scenario_2_student_email.py

# Custom request via command line
uv run python complex_examples/scenario_2_student_email.py "Email Jane Smith about her Science grade"

# Interactive mode (prompts for request)
uv run python complex_examples/scenario_2_student_email.py --interactive
uv run python complex_examples/scenario_2_student_email.py -i
```

**Example Requests:**
- "Send an email to John Doe about his performance"
- "Email S34567 regarding attendance"
- "Draft email to Bob Johnson about improvement"

**Checkpoints:**
1. **Checkpoint 1**: Review student performance → Decide to draft email (yes/no)
2. **Checkpoint 2**: Review email draft → Decide to send (yes/no)

**Possible Outcomes:**
- ✗ Cancel at checkpoint 1 → No email drafted
- 🗑️ Approve → Draft → Reject at checkpoint 2 → Email discarded
- ✓ Approve → Draft → Approve → Email sent

---

## Available Students

The mock database includes 3 students for testing:

| ID | Name | Performance | Use Case |
|----|------|-------------|----------|
| S12345 | John Doe | High performer (GPA 3.67) | Success scenarios |
| S23456 | Jane Smith | Good performer (GPA 3.45) | Balanced scenarios |
| S34567 | Bob Johnson | Needs support (GPA 2.89) | Intervention scenarios |

## Features

Both scripts include:
- ✓ Real LLM integration (requires OpenAI API key)
- ✓ Clear educational comments explaining LangGraph patterns
- ✓ Visual output with emojis and formatting
- ✓ Comprehensive error handling
- ✓ Command-line and interactive modes
- ✓ Graceful handling of Ctrl+C interrupts

## Learning Objectives

**After running these scripts, you'll understand:**

1. **ReAct Pattern** (Scenario 1)
   - How to implement iterative reasoning loops
   - Retry logic and error recovery strategies
   - Message accumulation for trace debugging
   - Conditional routing in cycles

2. **Human-in-the-Loop** (Scenario 2)
   - How to pause graph execution for user input
   - Implementing multiple decision checkpoints
   - Branching to different terminal nodes
   - State-based workflow control

3. **General LangGraph Concepts**
   - StateGraph construction with TypedDict
   - Node functions and state updates
   - Fixed vs conditional edges
   - Router functions for dynamic routing
   - Tool integration with `@tool` decorator
   - LLM integration with ChatOpenAI

## Troubleshooting

**"Error: Valid OPENAI_API_KEY required"**
- Add your OpenAI API key to `.env` file in parent directory
- Format: `OPENAI_API_KEY=sk-proj-...`

**Graph execution hangs (Scenario 2)**
- The script is waiting for your input at a checkpoint
- Type `yes` or `no` and press Enter

**"Student not found"**
- Use valid student IDs: S12345, S23456, or S34567
- Or use student names: John Doe, Jane Smith, Bob Johnson

**Import errors**
- Run `uv sync` from project root to install dependencies
- Ensure you're running from project root directory

## Advanced Usage

**Help information:**
```bash
uv run python complex_examples/scenario_1_student_performance.py --help
uv run python complex_examples/scenario_2_student_email.py --help
```

**Interrupt execution:**
- Press `Ctrl+C` at any time for graceful exit
- Scenario 2: Works during checkpoints as well

## File Structure

```
complex_examples/
├── README.md                           # This file
├── scenario_1_student_performance.py   # ReAct pattern demo (~600 lines)
└── scenario_2_student_email.py         # HITL pattern demo (~600 lines)
```

## Next Steps

- Modify the student database to add more students
- Extend the tool to query different data types
- Add more decision checkpoints to scenario 2
- Implement error recovery strategies in scenario 2
- Create your own scenarios combining these patterns

## Related Files

- `/examples/graph_helpers.py` - More graph pattern examples
- `/langgraph_deck.py` - Full interactive presentation
- `/README.md` - Main project documentation
