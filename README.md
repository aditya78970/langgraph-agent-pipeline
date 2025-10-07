# langgraph-agent-pipeline
AI agent for weather info and PDF document Q&amp;A using LangGraph &amp; LangChain

# LangGraph Agentic Pipeline - Weather & PDF RAG System

A sophisticated agentic pipeline built with LangGraph that intelligently routes queries to either fetch real-time weather data or answer questions from PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- **Intelligent Query Routing**: Automatically determines whether to fetch weather data or query PDF documents
- **Real-time Weather Data**: Integrates with OpenWeatherMap API for current weather information
- **PDF RAG System**: Process and query PDF documents using vector embeddings and semantic search
- **LangGraph Workflow**: Implements a state-based graph architecture for agent decision-making
- **Vector Database**: Uses Qdrant for efficient storage and retrieval of document embeddings
- **LangSmith Integration**: Complete observability and evaluation of LLM responses
- **Interactive UI**: Streamlit-based chat interface for easy interaction
- **Comprehensive Testing**: Unit tests for all major components

##  Architecture

```
User Query → Route Decision Node → [Weather API | PDF RAG] → LLM Processing → Response
                                    ↓                ↓
                              OpenWeather API    Qdrant Vector DB
                                                 (PDF Embeddings)
```

### LangGraph Workflow

The system uses a state-based graph with the following nodes:

1. **Route Node**: Analyzes the query and decides the appropriate path
2. **Weather Node**: Fetches real-time weather data from OpenWeatherMap API
3. **PDF Node**: Retrieves relevant context from PDF using vector similarity search
4. **Response Node**: Generates natural language responses using GPT-4

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- OpenWeatherMap API key
- LangSmith API key (optional, for tracing)

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/langgraph-agent-pipeline.git
cd langgraph-agent-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
OPENAI_API_KEY=sk-...
OPENWEATHER_API_KEY=your_key_here
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph-weather-rag
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- OpenWeatherMap: https://openweathermap.org/api
- LangSmith: https://smith.langchain.com/

### 5. Create Required Directories

```bash
mkdir documents
mkdir qdrant_db
```

## 📖 Usage

### Running the Streamlit UI

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload a PDF**: Use the sidebar to upload a PDF document
2. **Process the PDF**: Click "Process PDF" to create embeddings
3. **Ask Questions**: Type queries in the chat interface

**Example Queries:**
- Weather: "What's the weather in London?"
- PDF: "What is mentioned in the document about AI?"

### Running Tests

```bash
python test_agent.py
```

Or with pytest:

```bash
pytest test_agent.py -v
```

### Command Line Usage

```python
from main import run_agent

# Weather query
result = run_agent("What's the temperature in Tokyo?")
print(result['final_response'])

# PDF query
result = run_agent("Summarize the main points from the document")
print(result['final_response'])
```

## 🔧 Project Structure

```
langgraph-agent-pipeline/
├── main.py                 # Core LangGraph agent implementation
├── app.py                  # Streamlit UI application
├── test_agent.py          # Unit tests
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── README.md             # This file
├── documents/            # Directory for PDF uploads
│   └── sample.pdf
├── qdrant_db/           # Qdrant vector database storage
└── screenshots/         # LangSmith screenshots
    ├── trace_weather.png
    ├── trace_pdf.png
    └── evaluation.png
```

## 🧪 Implementation Details

### 1. LangGraph State Management

The agent uses a typed state dictionary:

```python
class AgentState(TypedDict):
    query: str                  # User's input query
    route: Literal["weather", "pdf", "unknown"]  # Routing decision
    weather_data: dict          # Weather API response
    pdf_context: str            # Retrieved PDF context
    final_response: str         # Generated response
    messages: list              # Execution trace
```

### 2. Weather Service Integration

- Uses OpenWeatherMap API for real-time data
- Extracts city names using LLM
- Returns temperature, humidity, wind speed, and conditions
- Handles API errors gracefully

### 3. PDF RAG Pipeline

**Document Processing:**
1. Load PDF using PyPDFLoader
2. Split into chunks (1000 chars, 200 overlap)
3. Generate embeddings with OpenAI
4. Store in Qdrant vector database

**Query Processing:**
1. Convert query to embedding
2. Perform similarity search (top 3 results)
3. Combine retrieved chunks as context
4. Generate response using LLM

### 4. LLM Processing

- Model: GPT-4o-mini (configurable)
- Temperature: 0 (deterministic responses)
- Prompts optimized for each task type
- Contextual responses based on retrieved data

### 5. Vector Database (Qdrant)

- Local storage mode (no server required)
- Cosine similarity for document matching
- Collection per PDF document
- 1536-dimensional vectors (OpenAI embeddings)

## 📊 LangSmith Integration

### Viewing Traces

1. Visit https://smith.langchain.com/
2. Navigate to your project: `langgraph-weather-rag`
3. View detailed traces for each query

### Trace Information Includes:

- Node execution order
- Input/output at each step
- LLM token usage
- Latency metrics
- Error tracking

### Example Trace Flow:

```
Query: "What's the weather in Paris?"
├─ route_query: 0.15s
│  └─ Decision: weather
├─ fetch_weather: 1.23s
│  ├─ City Extraction (LLM): 0.45s
│  └─ Weather API Call: 0.78s
└─ generate_response: 0.89s
   └─ Response Generation (LLM): 0.89s
Total: 2.27s
```

## Testing

### Test Coverage

The test suite includes:

1. **API Handling Tests**
   - Successful weather API calls
   - Error handling
   - Timeout scenarios

2. **PDF Processing Tests**
   - Document loading
   - Embedding generation
   - Similarity search

3. **LangGraph Node Tests**
   - Query routing logic
   - Weather fetching
   - PDF querying
   - Response generation

4. **Integration Tests**
   - End-to-end workflow
   - State transitions
   - Error propagation

### Running Specific Tests

```bash
# Run specific test class
python -m unittest test_agent.TestWeatherService

# Run with coverage
pip install coverage
coverage run -m unittest test_agent
coverage report
```

## Key Features Demonstrated

### LangGraph Implementation
- State-based workflow management
- Conditional edge routing
- Node-based processing
- Clean separation of concerns

### Real-time API Integration
- OpenWeatherMap API calls
- Error handling and retries
- Response parsing and formatting

### ✅ RAG System
- PDF document processing
- Vector embedding generation
- Semantic similarity search
- Context-aware responses

### ✅ LLM Processing
- OpenAI GPT-4 integration
- Prompt engineering
- Temperature control
- Response formatting

### ✅ Vector Database
- Qdrant integration
- Efficient storage and retrieval
- Collection management
- Similarity search optimization

### ✅ Observability
- LangSmith tracing
- Execution monitoring
- Performance metrics
- Error tracking

### ✅ Testing
- Unit tests for all components
- Mock-based testing
- Integration tests
- High code coverage

### ✅ User Interface
- Streamlit chat interface
- PDF upload functionality
- Real-time responses
- Debug information display

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade langchain langgraph langchain-openai
```

**2. API Key Issues**
- Verify `.env` file exists and contains valid keys
- Check API key permissions and quotas
- Ensure no extra spaces in environment variables

**3. Qdrant Connection Issues**
```bash
rm -rf qdrant_db/
# Restart the application to recreate the database
```

**4. PDF Processing Errors**
- Ensure PDF is not corrupted
- Check file size (recommend < 10MB)
- Verify PDF contains extractable text

## 📈 Performance Considerations

- **Embedding Generation**: ~2-3 seconds for typical PDF
- **Weather API Call**: ~0.5-1 second
- **RAG Query**: ~1-2 seconds (includes similarity search + LLM)
- **Average Response Time**: 2-3 seconds end-to-end

## 🔒 Security Notes

- Never commit `.env` file to version control
- Rotate API keys regularly
- Use environment variables for all secrets
- Implement rate limiting for production use

## 🚧 Future Enhancements

- [ ] Support multiple PDF documents
- [ ] Add memory/conversation history
- [ ] Implement streaming responses
- [ ] Add more weather providers
- [ ] Support additional document formats (DOCX, TXT)
- [ ] Add authentication to Streamlit app
- [ ] Deploy to cloud platform
- [ ] Add caching for repeated queries
- [ ] Implement async processing
- [ ] Add support for images in PDFs

## 📝 License

MIT License - feel free to use this project for learning and development.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with**: LangGraph • LangChain • OpenAI • Qdrant • Streamlit • LangSmith
