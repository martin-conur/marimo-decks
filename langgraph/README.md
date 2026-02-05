# LangGraph Educational Hub

A comprehensive collection of interactive LangGraph tutorials and examples using marimo. Learn how to build stateful AI agents with graphs, from fundamental concepts to advanced patterns with hands-on examples.

## What's Included

### 📊 Interactive Presentation (`langgraph_deck.py`)
- **14 Interactive Slides** covering LangGraph from basics to advanced patterns
- **Live Code Examples** with reactive updates as you change inputs
- **Mock Mode** for offline demonstrations (no API keys required)
- **Real-World Demo** showing a complete data analysis pipeline with LLM-generated insights
- **LangSmith Integration** for execution tracing and debugging

### 🎯 Complex Examples (`complex_examples/`)
- **Standalone Scripts** demonstrating real-world LangGraph patterns
- **Scenario 1**: ReAct pattern with retry logic and error recovery
- **Scenario 2**: Human-in-the-loop workflow with conditional routing
- **Terminal-Based** - Run from command line with custom questions
- **Production-Ready Patterns** - Real LLM integration (no mock mode)

## What You'll Learn

### From the Interactive Presentation:
1. **Core Concepts**: Graphs, nodes, edges, and state management
2. **LLM Integration**: Adding AI capabilities to your workflows
3. **Conditional Routing**: Building decision trees in your graphs
4. **Tools & Function Calling**: Extending agent capabilities
5. **Cycles & Loops**: Implementing ReAct patterns
6. **Real-World Application**: Complete data analysis pipeline

### From Complex Examples:
1. **ReAct Pattern**: Iterative reasoning with automatic retry logic
2. **Human-in-the-Loop**: Multi-checkpoint workflows with user decisions
3. **Error Recovery**: Handling failures gracefully with retries
4. **Branching Workflows**: Conditional routing to multiple outcomes
5. **Production Patterns**: Real LLM integration without mock modes
6. **Command-Line Tools**: Building executable LangGraph scripts

## Prerequisites

- Python 3.10 or higher
- Basic knowledge of Python and AI/LLM concepts

## Setup

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Navigate

```bash
cd langgraph
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment (Optional)

For full LLM features, set up your API keys:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Note**:
- The **presentation** works in mock mode without API keys for learning and testing
- The **complex examples** require a valid OpenAI API key to run

## Quick Start

### Option 1: Interactive Presentation (Recommended for Learning)

**Presentation Mode (Slides):**
```bash
uv run marimo run langgraph_deck.py
```

This launches the interactive presentation in your browser. Navigate with arrow keys or on-screen controls.

**Edit Mode (Development):**
```bash
uv run marimo edit langgraph_deck.py
```

Use this mode to explore the code, modify examples, and experiment with different approaches.

### Option 2: Complex Examples (Hands-On Practice)

**Scenario 1 - ReAct Pattern:**
```bash
# Default example
uv run python complex_examples/scenario_1_student_performance.py

# Custom question
uv run python complex_examples/scenario_1_student_performance.py "What is Jane Smith's GPA?"

# Interactive mode
uv run python complex_examples/scenario_1_student_performance.py --interactive
```

**Scenario 2 - Human-in-the-Loop:**
```bash
# Default example (requires user input at 2 checkpoints)
uv run python complex_examples/scenario_2_student_email.py

# Custom request
uv run python complex_examples/scenario_2_student_email.py "Email Bob Johnson about attendance"

# Interactive mode
uv run python complex_examples/scenario_2_student_email.py --interactive
```

📖 **See [`complex_examples/README.md`](complex_examples/README.md) for detailed documentation**

## Presentation Structure

1. **Introduction** - What is LangGraph and why it matters
2. **State Concepts** - Understanding shared memory in graphs
3. **Nodes** - Functions that transform state
4. **Edges** - Control flow and routing
5. **First Graph** - Build a simple linear workflow
6. **LLM Integration** - Adding AI to your nodes
7. **Conditional Routing** - Dynamic decision making
8. **Tools & Function Calling** - Extending capabilities
9. **Cycles & Loops** - ReAct pattern explained
10. **Data Pipeline Architecture** - Real-world example overview
11. **Data Pipeline Implementation** - Complete working demo
12. **Observability** - LangSmith tracing and debugging
13. **Advanced Patterns** - Multi-agent, subgraphs, and more
14. **Summary** - Key takeaways and next steps

## Interactive Features

- **Mock Mode Toggle**: Run examples without API calls
- **Text Inputs**: Change questions and see AI responses
- **Dropdowns**: Select different routing options
- **Sliders**: Adjust parameters like max iterations
- **File Upload**: Try the data pipeline with your own CSV files
- **Real-time Visualization**: See graphs and charts update instantly

## Data Analysis Pipeline Demo

The presentation includes a complete data analysis workflow:

1. **Load Data**: Upload CSV or use sample sales data
2. **Analyze**: Calculate statistics with pandas
3. **Generate Insights**: LLM interprets patterns and provides recommendations
4. **Visualize**: Create interactive charts with plotly

Try it with your own data or use the included sample dataset!

## Recommended Learning Path

1. **Start with the Presentation** - Get foundational understanding
   ```bash
   uv run marimo run langgraph_deck.py
   ```
   - Work through slides 1-14
   - Experiment with mock mode first
   - Enable real LLM calls later

2. **Practice with Complex Examples** - Apply concepts hands-on
   ```bash
   uv run python complex_examples/scenario_1_student_performance.py --interactive
   uv run python complex_examples/scenario_2_student_email.py --interactive
   ```
   - Try different questions
   - Observe reasoning traces
   - Experience human-in-the-loop workflows

3. **Explore the Code** - Understand implementation details
   ```bash
   uv run marimo edit langgraph_deck.py
   ```
   - Read `examples/graph_helpers.py`
   - Modify complex examples
   - Build your own scenarios

## Troubleshooting

### Import Errors

If you see import errors, ensure dependencies are installed:

```bash
uv sync
```

### API Key Issues

**For the Presentation:**
- Defaults to mock mode (no API key needed)
- To use real LLM features: add `OPENAI_API_KEY` to `.env` and toggle off "Mock mode"

**For Complex Examples:**
- Requires valid `OPENAI_API_KEY` in `.env` file
- Will exit with error message if key is missing
- Format: `OPENAI_API_KEY=sk-proj-...`

### Port Already in Use (Presentation)

If port 2718 is busy, marimo will automatically try another port.

### Browser Doesn't Open (Presentation)

Manually open the URL shown in the terminal (usually `http://localhost:2718`).

### Script Hangs (Complex Examples)

**Scenario 2 pauses at checkpoints waiting for your input:**
- Type `yes` or `no` and press Enter
- Press `Ctrl+C` to exit gracefully at any time

## Project Structure

```
langgraph/
├── langgraph_deck.py              # Main presentation (14 slides)
├── pyproject.toml                 # UV project configuration
├── .env.example                   # API keys template
├── README.md                      # This file
├── examples/                      # Utilities for presentation
│   ├── __init__.py
│   ├── graph_helpers.py           # LangGraph graph builders
│   ├── visualization.py           # Plotly charts and rendering
│   └── sample_data.csv            # Demo sales dataset
└── complex_examples/              # Standalone scripts
    ├── README.md                  # Complex examples documentation
    ├── scenario_1_student_performance.py   # ReAct pattern demo
    └── scenario_2_student_email.py         # HITL pattern demo
```

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Marimo Documentation](https://docs.marimo.io/)
- [LangSmith Platform](https://smith.langchain.com/)

## Contributing

This is an educational project. Feel free to adapt and extend it for your own learning or teaching purposes.

## License

MIT License - See LICENSE file for details
