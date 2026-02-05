# LangGraph Marimo Deck - Quick Start

## ✅ Everything is Ready!

Your LangGraph educational presentation is complete with:
- ✓ 24 interactive cells (14 slides)
- ✓ All dependencies installed
- ✓ API keys configured
- ✓ Sample data loaded
- ✓ No errors or conflicts

## 🚀 Run the Presentation

### Presentation Mode (Slides)
```bash
uv run marimo run langgraph_deck.py
```

Opens at: `http://localhost:2718`

### Edit Mode (Development)
```bash
uv run marimo edit langgraph_deck.py
```

## 🎯 Interactive Features with Real API

With your OpenAI and LangSmith keys configured, you can:

### Slide 6: LLM Integration
- **Uncheck "Mock mode"**
- Ask any question
- Get real GPT-4 responses

### Slide 7: Conditional Routing
- **Uncheck "Mock mode"**
- Try both math and general questions
- See real AI routing

### Slide 11: Data Analysis Pipeline (★ Best Demo!)
- **Uncheck "Mock mode"**
- Upload your own CSV or use sample data
- Get AI-powered insights and visualizations
- **CSV format**: `date,product,region,revenue,quantity`

## 📊 LangSmith Tracing

Every LLM call is automatically traced to [smith.langchain.com](https://smith.langchain.com):
- Project: **langgraph-presentation**
- View execution traces
- Track token usage and costs
- Debug your workflows

## 🎨 Navigation

- **Arrow Keys**: ← → to navigate slides
- **Mouse**: Click on-screen arrows
- **Escape**: Exit fullscreen
- **Interactive Elements**: All inputs work during presentation

## 📝 What's Included

### Slides 1-5: Fundamentals
- Introduction to LangGraph
- State management (TypedDict vs MessagesState)
- Nodes as functions
- Edges (fixed and conditional)
- First graph (3-node linear workflow)

### Slides 6-9: Intelligence
- LLM integration with OpenAI
- Conditional routing (math vs general)
- Tools and function calling
- ReAct pattern (reason → act → observe)

### Slides 10-11: Real-World Demo
- Data analysis pipeline architecture
- **Full working implementation** with:
  - Data loading (CSV)
  - Statistical analysis (pandas)
  - AI insights (GPT-4)
  - Interactive visualizations (plotly)

### Slides 12-14: Advanced & Resources
- LangSmith observability
- Advanced patterns (subgraphs, multi-agent, etc.)
- Summary and next steps

## 🛠️ Troubleshooting

### Port Already in Use
Marimo will automatically try another port or specify:
```bash
uv run marimo run langgraph_deck.py --port 8080
```

### API Key Issues
1. Check `.env` file exists: `cat .env`
2. Verify keys don't start with placeholder text
3. Restart presentation after changing `.env`

### Import Errors
```bash
uv sync
```

## 💡 Pro Tips

1. **Start in Mock Mode**: Explore without API costs
2. **Toggle Strategically**: Only disable mock mode when ready
3. **Upload Custom Data**: Slide 11 accepts any CSV with the right columns
4. **Watch LangSmith**: Open dashboard before running to see traces live
5. **Experiment**: Change inputs, parameters, and see reactive updates

## 📦 What's Next?

After exploring the presentation:
1. **Modify examples**: Edit `langgraph_deck.py` with your own use cases
2. **Build your graph**: Start with the examples and extend them
3. **Add slides**: Follow the cell pattern (UI cell + content cell)
4. **Share**: Present to your team or students!

---

**Enjoy your LangGraph presentation! 🎉**

Questions? Check the main [README.md](README.md) for detailed documentation.
