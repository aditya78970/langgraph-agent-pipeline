#----------------claude version

import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import operator

# State definition
class AgentState(TypedDict):
    query: str
    route: Literal["weather", "pdf", "unknown"]
    weather_data: dict
    pdf_context: str
    final_response: str
    messages: Annotated[list, operator.add]

# Configuration
class Config:
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langgraph-weather-rag")
    QDRANT_PATH = "./qdrant_db"
    PDF_PATH = "./documents/sample.pdf"

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# Initialize Qdrant client
qdrant_client = QdrantClient(path=Config.QDRANT_PATH)

class WeatherService:
    """Service for fetching weather data"""
    
    @staticmethod
    def get_weather(city: str) -> dict:
        """Fetch weather data from OpenWeatherMap API"""
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": Config.OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

class PDFProcessor:
    """Service for processing PDF documents"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.collection_name = "pdf_documents"
        
    def load_and_process_pdf(self):
        """Load PDF and create embeddings"""
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create collection if not exists
        try:
            qdrant_client.get_collection(self.collection_name)
        except:
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        # Create vector store
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=self.collection_name,
            embeddings=embeddings
        )
        
        # Add documents
        vectorstore.add_documents(splits)
        return vectorstore
    
    def query_pdf(self, query: str, k: int = 3) -> str:
        """Query the PDF using RAG"""
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=self.collection_name,
            embeddings=embeddings
        )
        
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=k)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

# LangGraph Nodes
def route_query(state: AgentState) -> AgentState:
    """Decide whether to call weather API or query PDF"""
    query = state["query"].lower()
    
    # Simple routing logic
    weather_keywords = ["weather", "temperature", "forecast", "climate", "rain", "sunny"]
    
    if any(keyword in query for keyword in weather_keywords):
        state["route"] = "weather"
    else:
        state["route"] = "pdf"
    
    state["messages"] = [f"Routed query to: {state['route']}"]
    return state

def fetch_weather(state: AgentState) -> AgentState:
    """Fetch weather data"""
    if state["route"] != "weather":
        return state
    
    # Extract city from query (simple approach)
    prompt = ChatPromptTemplate.from_template(
        "Extract the city name from this query. Return only the city name, nothing else: {query}"
    )
    city_response = llm.invoke(prompt.format_messages(query=state["query"]))
    city = city_response.content.strip()
    
    # Fetch weather
    weather_data = WeatherService.get_weather(city)
    state["weather_data"] = weather_data
    state["messages"].append(f"Fetched weather data for: {city}")
    
    return state

def query_pdf_node(state: AgentState) -> AgentState:
    """Query PDF using RAG"""
    if state["route"] != "pdf":
        return state
    
    processor = PDFProcessor(Config.PDF_PATH)
    context = processor.query_pdf(state["query"])
    state["pdf_context"] = context
    state["messages"].append("Retrieved relevant context from PDF")
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate final response using LLM"""
    if state["route"] == "weather":
        # Format weather response
        weather = state.get("weather_data", {})
        if "error" in weather:
            response = f"Error fetching weather data: {weather['error']}"
        else:
            prompt = ChatPromptTemplate.from_template(
                """Based on the following weather data, provide a natural language response to the user's query.
                
                Query: {query}
                
                Weather Data:
                - City: {city}
                - Temperature: {temp}°C
                - Feels Like: {feels_like}°C
                - Weather: {description}
                - Humidity: {humidity}%
                - Wind Speed: {wind_speed} m/s
                
                Provide a helpful and conversational response."""
            )
            
            response = llm.invoke(prompt.format_messages(
                query=state["query"],
                city=weather.get("name", "Unknown"),
                temp=weather.get("main", {}).get("temp", "N/A"),
                feels_like=weather.get("main", {}).get("feels_like", "N/A"),
                description=weather.get("weather", [{}])[0].get("description", "N/A"),
                humidity=weather.get("main", {}).get("humidity", "N/A"),
                wind_speed=weather.get("wind", {}).get("speed", "N/A")
            ))
            response = response.content
    
    elif state["route"] == "pdf":
        # Format PDF response
        context = state.get("pdf_context", "")
        if not context:
            response = "No relevant information found in the PDF."
        else:
            prompt = ChatPromptTemplate.from_template(
                """Based on the following context from the PDF document, answer the user's query.
                
                Query: {query}
                
                Context:
                {context}
                
                Provide a clear and concise answer based only on the information in the context."""
            )
            
            response = llm.invoke(prompt.format_messages(
                query=state["query"],
                context=context
            ))
            response = response.content
    else:
        response = "I couldn't determine how to handle your query."
    
    state["final_response"] = response
    state["messages"].append("Generated final response")
    return state

# Build the graph
def build_agent_graph():
    """Build the LangGraph agent"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route_node", route_query)
    workflow.add_node("weather", fetch_weather)
    workflow.add_node("pdf", query_pdf_node)
    workflow.add_node("respond", generate_response)
    
    # Add edges
    workflow.set_entry_point("route_node")
    
    # Conditional edges from route
    workflow.add_conditional_edges(
        "route_node",
        lambda x: x["route"],
        {
            "weather": "weather",
            "pdf": "pdf"
        }
    )
    
    workflow.add_edge("weather", "respond")
    workflow.add_edge("pdf", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()

# Main execution function
def run_agent(query: str) -> dict:
    """Run the agent with a query"""
    app = build_agent_graph()
    
    initial_state = {
        "query": query,
        "route": "unknown",
        "weather_data": {},
        "pdf_context": "",
        "final_response": "",
        "messages": []
    }
    
    result = app.invoke(initial_state)
    return result

if __name__ == "__main__":
    # Example usage
    print("Testing Weather Query:")
    result1 = run_agent("What's the weather like in London?")
    print(f"Response: {result1['final_response']}\n")
    
    print("Testing PDF Query:")
    result2 = run_agent("What is mentioned in the document?")
    print(f"Response: {result2['final_response']}")

