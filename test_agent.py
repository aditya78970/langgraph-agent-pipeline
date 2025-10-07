import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    WeatherService, 
    PDFProcessor, 
    route_query, 
    fetch_weather,
    query_pdf_node,
    generate_response,
    AgentState
)

class TestWeatherService(unittest.TestCase):
    """Test cases for Weather API handling"""
    
    @patch('main.requests.get')
    def test_get_weather_success(self, mock_get):
        """Test successful weather API call"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "London",
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 72
            },
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 3.5}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = WeatherService.get_weather("London")
        
        self.assertEqual(result["name"], "London")
        self.assertEqual(result["main"]["temp"], 15.5)
        mock_get.assert_called_once()
    
    @patch('main.requests.get')
    def test_get_weather_api_error(self, mock_get):
        """Test weather API error handling"""
        mock_get.side_effect = Exception("API Error")
        
        result = WeatherService.get_weather("InvalidCity")
        
        self.assertIn("error", result)
    
    @patch('main.requests.get')
    def test_get_weather_timeout(self, mock_get):
        """Test weather API timeout"""
        mock_get.side_effect = TimeoutError("Request timeout")
        
        result = WeatherService.get_weather("London")
        
        self.assertIn("error", result)

class TestPDFProcessor(unittest.TestCase):
    """Test cases for PDF processing and RAG"""
    
    @patch('main.PyPDFLoader')
    @patch('main.qdrant_client')
    def test_load_and_process_pdf(self, mock_qdrant, mock_loader):
        """Test PDF loading and embedding creation"""
        # Mock PDF loader
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_loader.return_value.load.return_value = [mock_doc]
        
        processor = PDFProcessor("test.pdf")
        
        # This would normally process the PDF
        # We're testing that it doesn't crash
        self.assertEqual(processor.pdf_path, "test.pdf")
    
    @patch('main.Qdrant')
    def test_query_pdf(self, mock_qdrant):
        """Test PDF querying with RAG"""
        # Mock vector store
        mock_vectorstore = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a test document about AI."
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        mock_qdrant.return_value = mock_vectorstore
        
        processor = PDFProcessor("test.pdf")
        result = processor.query_pdf("What is AI?")
        
        self.assertIsInstance(result, str)
        self.assertIn("This is a test document", result)

class TestLangGraphNodes(unittest.TestCase):
    """Test cases for LangGraph node functions"""
    
    def test_route_query_weather(self):
        """Test routing to weather service"""
        state = AgentState(
            query="What's the weather in Paris?",
            route="unknown",
            weather_data={},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = route_query(state)
        
        self.assertEqual(result["route"], "weather")
        self.assertIn("Routed query to", result["messages"][0])
    
    def test_route_query_pdf(self):
        """Test routing to PDF service"""
        state = AgentState(
            query="What is mentioned in the document?",
            route="unknown",
            weather_data={},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = route_query(state)
        
        self.assertEqual(result["route"], "pdf")
    
    @patch('main.WeatherService.get_weather')
    @patch('main.llm')
    def test_fetch_weather_node(self, mock_llm, mock_weather):
        """Test weather fetching node"""
        # Mock LLM response for city extraction
        mock_response = MagicMock()
        mock_response.content = "London"
        mock_llm.invoke.return_value = mock_response
        
        # Mock weather service
        mock_weather.return_value = {
            "name": "London",
            "main": {"temp": 15}
        }
        
        state = AgentState(
            query="What's the weather in London?",
            route="weather",
            weather_data={},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = fetch_weather(state)
        
        self.assertIn("weather_data", result)
        self.assertEqual(result["weather_data"]["name"], "London")
    
    def test_fetch_weather_skip_if_not_routed(self):
        """Test that weather fetch is skipped if not routed to weather"""
        state = AgentState(
            query="What is in the PDF?",
            route="pdf",
            weather_data={},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = fetch_weather(state)
        
        # State should remain unchanged
        self.assertEqual(result["route"], "pdf")
        self.assertEqual(result["weather_data"], {})

class TestResponseGeneration(unittest.TestCase):
    """Test cases for LLM response generation"""
    
    @patch('main.llm')
    def test_generate_weather_response(self, mock_llm):
        """Test weather response generation"""
        mock_response = MagicMock()
        mock_response.content = "The weather in London is cloudy with a temperature of 15°C."
        mock_llm.invoke.return_value = mock_response
        
        state = AgentState(
            query="What's the weather in London?",
            route="weather",
            weather_data={
                "name": "London",
                "main": {"temp": 15, "feels_like": 14, "humidity": 72},
                "weather": [{"description": "cloudy"}],
                "wind": {"speed": 3.5}
            },
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = generate_response(state)
        
        self.assertIn("final_response", result)
        self.assertIsInstance(result["final_response"], str)
        self.assertTrue(len(result["final_response"]) > 0)
    
    @patch('main.llm')
    def test_generate_pdf_response(self, mock_llm):
        """Test PDF RAG response generation"""
        mock_response = MagicMock()
        mock_response.content = "According to the document, AI is artificial intelligence."
        mock_llm.invoke.return_value = mock_response
        
        state = AgentState(
            query="What is AI?",
            route="pdf",
            weather_data={},
            pdf_context="Artificial Intelligence (AI) is the simulation of human intelligence.",
            final_response="",
            messages=[]
        )
        
        result = generate_response(state)
        
        self.assertIn("final_response", result)
        self.assertTrue(len(result["final_response"]) > 0)
    
    def test_generate_response_with_error(self):
        """Test response generation with weather error"""
        state = AgentState(
            query="What's the weather?",
            route="weather",
            weather_data={"error": "API Error"},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        result = generate_response(state)
        
        self.assertIn("Error", result["final_response"])

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    @patch('main.WeatherService.get_weather')
    @patch('main.llm')
    def test_full_weather_pipeline(self, mock_llm, mock_weather):
        """Test complete weather query pipeline"""
        # Mock city extraction
        city_response = MagicMock()
        city_response.content = "Paris"
        
        # Mock final response
        final_response = MagicMock()
        final_response.content = "It's sunny in Paris with 20°C."
        
        mock_llm.invoke.side_effect = [city_response, final_response]
        
        mock_weather.return_value = {
            "name": "Paris",
            "main": {"temp": 20, "feels_like": 19, "humidity": 60},
            "weather": [{"description": "sunny"}],
            "wind": {"speed": 2.5}
        }
        
        # Test routing
        state = AgentState(
            query="What's the weather in Paris?",
            route="unknown",
            weather_data={},
            pdf_context="",
            final_response="",
            messages=[]
        )
        
        state = route_query(state)
        self.assertEqual(state["route"], "weather")
        
        state = fetch_weather(state)
        self.assertIn("name", state["weather_data"])
        
        state = generate_response(state)
        self.assertTrue(len(state["final_response"]) > 0)

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    run_tests()
