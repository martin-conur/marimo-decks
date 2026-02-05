# LangGraph Educational Presentation

An interactive presentation about LangGraph using marimo slides. Learn how to build stateful AI agents with graphs, from fundamental concepts to a real-world data analysis pipeline.

## Features

- **14 Interactive Slides** covering LangGraph from basics to advanced patterns
- **Live Code Examples** with reactive updates as you change inputs
- **Mock Mode** for offline demonstrations (no API keys required)
- **Real-World Demo** showing a complete data analysis pipeline with LLM-generated insights
- **LangSmith Integration** for execution tracing and debugging

## What You'll Learn

1. **Core Concepts**: Graphs, nodes, edges, and state management
2. **LLM Integration**: Adding AI capabilities to your workflows
3. **Conditional Routing**: Building decision trees in your graphs
4. **Tools & Function Calling**: Extending agent capabilities
5. **Cycles & Loops**: Implementing ReAct patterns
6. **Real-World Application**: Complete data analysis pipeline

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

**Note**: The presentation works in mock mode without API keys for learning and testing.

## Running the Presentation

### Presentation Mode (Slides)

```bash
uv run marimo run langgraph_deck.py
```

This launches the interactive presentation in your browser. Navigate with arrow keys or on-screen controls.

### Edit Mode (Development)

```bash
uv run marimo edit langgraph_deck.py
```

Use this mode to explore the code, modify examples, and experiment with different approaches.

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

## Troubleshooting

### Import Errors

If you see import errors, ensure dependencies are installed:

```bash
uv sync
```

### API Key Issues

The presentation defaults to mock mode. To use real LLM features:

1. Ensure `.env` file exists with valid `OPENAI_API_KEY`
2. Toggle off "Mock mode" checkboxes in slides 6, 7, and 11

### Port Already in Use

If port 2718 is busy, marimo will automatically try another port.

### Browser Doesn't Open

Manually open the URL shown in the terminal (usually `http://localhost:2718`).

## Project Structure

```
langgraph/
├── langgraph_deck.py       # Main presentation (14 slides)
├── pyproject.toml           # UV project configuration
├── .env.example             # API keys template
├── README.md                # This file
└── examples/
    ├── __init__.py
    ├── graph_helpers.py     # LangGraph utilities
    ├── visualization.py     # Rendering and charts
    └── sample_data.csv      # Demo dataset
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
