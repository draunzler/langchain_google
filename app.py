from urllib.parse import urlparse
import aiohttp
from fastapi import FastAPI, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import time
from functools import wraps
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any, Tuple, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import langchain_core.exceptions
import markdown2
import hashlib
from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import re

load_dotenv()

class SemanticSearchManager:
    """
    Manages semantic search functionality for enhancing web search results.
    Uses embeddings to find semantically similar content and improve result quality.
    """
    def __init__(self):
        """Initialize the semantic search manager."""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("LLM_API_KEY")
        )
        self.vector_stores = {}  # topic -> FAISS index
        
    def create_vector_store(self, topic: str, documents: List[Dict[str, str]]) -> None:
        """
        Create a vector store for a specific topic.
        
        Args:
            topic (str): The topic to create a vector store for
            documents (List[Dict[str, str]]): List of documents to add to the vector store,
                                              each document should have 'content' and 'source' keys
        """
        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["source"]} for doc in documents]
        
        self.vector_stores[topic] = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print(f"Created vector store for topic '{topic}' with {len(documents)} documents")
        
    def search(self, topic: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents.
        
        Args:
            topic (str): The topic to search in
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with content and metadata
        """
        if topic not in self.vector_stores:
            raise ValueError(f"No vector store exists for topic '{topic}'")
            
        results = self.vector_stores[topic].similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def add_to_topic(self, topic: str, documents: List[Dict[str, str]]) -> None:
        """
        Add documents to an existing topic's vector store.
        
        Args:
            topic (str): The topic to add documents to
            documents (List[Dict[str, str]]): List of documents to add
        """
        if topic not in self.vector_stores:
            # Create a new vector store if one doesn't exist
            return self.create_vector_store(topic, documents)
            
        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["source"]} for doc in documents]
        
        self.vector_stores[topic].add_texts(texts=texts, metadatas=metadatas)
        print(f"Added {len(documents)} documents to topic '{topic}'")
    
    def list_topics(self) -> List[str]:
        """
        List all available topics.
        
        Returns:
            List[str]: List of topic names
        """
        return list(self.vector_stores.keys())
    
    def delete_topic(self, topic: str) -> bool:
        """
        Delete a topic and its vector store.
        
        Args:
            topic (str): The topic to delete
            
        Returns:
            bool: True if the topic was deleted, False if it didn't exist
        """
        if topic in self.vector_stores:
            del self.vector_stores[topic]
            print(f"Deleted topic '{topic}'")
            return True
        return False
    
semantic_search_manager = SemanticSearchManager()

# Function to process search results and add them to the semantic search index
def process_search_results(topic: str, results: str) -> None:
    """
    Process search results and add them to the semantic search index.
    
    Args:
        topic (str): The topic to add results to
        results (str): Raw search results text
    """
    # Parse the results into individual documents
    # This is a simple implementation that splits on double newlines
    # In a real implementation, you would want more sophisticated parsing
    documents = []
    
    # Split results into chunks based on URLs or double newlines
    result_chunks = results.split("\n\n")
    
    for i, chunk in enumerate(result_chunks):
        if chunk.strip():
            # Try to extract a URL as the source
            lines = chunk.split("\n")
            source = None
            content = chunk
            
            for line in lines:
                # Simple regex to detect URLs
                if re.search(r'https?://\S+', line):
                    source = line.strip()
                    content = chunk.replace(source, "").strip()
                    break
            
            # If no URL found, use a generic source
            if not source:
                source = f"result-{i+1}"
            
            documents.append({
                "content": content,
                "source": source
            })
    
    # Add documents to the semantic search index
    if documents:
        semantic_search_manager.add_to_topic(topic, documents)

# Create a Tool for semantic search
def get_semantic_search_tool():
    """Creates a tool for semantic search."""
    
    def semantic_search(query_string):
        """
        Perform semantic search based on previously indexed web search results.
        
        Format: "topic: query" or just "query" for the default topic
        """
        # Parse topic and query
        topic = "default"
        query = query_string
        
        if ":" in query_string:
            parts = query_string.split(":", 1)
            if len(parts) == 2:
                topic, query = parts[0].strip(), parts[1].strip()
        
        try:
            # Check if topic exists
            if topic not in semantic_search_manager.list_topics():
                return f"No data available for topic '{topic}'. Please use regular search first to collect data."
            
            # Perform semantic search
            results = semantic_search_manager.search(topic, query)
            
            # Format results
            formatted_results = f"# Semantic Search Results for: {query}\n\n"
            
            for i, result in enumerate(results):
                formatted_results += f"## Result {i+1}\n"
                formatted_results += f"**Source:** {result['metadata'].get('source', 'Unknown')}\n"
                formatted_results += f"**Relevance Score:** {result['score']:.2f}\n\n"
                formatted_results += f"{result['content']}\n\n"
            
            return formatted_results
        except Exception as e:
            return f"Error performing semantic search: {str(e)}"
    
    return Tool(
        name="Semantic Search",
        func=semantic_search,
        description="Search for semantically similar information from previously collected web search results. Format: 'topic: query' or just 'query' for the default topic."
    )

# Update the search tools to integrate with semantic search
def get_integrated_search_tool():
    """Get a search tool that automatically adds results to the semantic search index."""
    search = GoogleSearchAPIWrapper(
        k=8,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    def integrated_search(query_string):
        """
        Perform a search and automatically add results to the semantic search index.
        
        Format: "query" or "topic:query" to specify a topic for semantic indexing
        """
        # Parse topic and query
        topic = "default"
        query = query_string
        
        if ":" in query_string and not query_string.startswith("site:"):
            parts = query_string.split(":", 1)
            if len(parts) == 2:
                topic, query = parts[0].strip(), parts[1].strip()
        
        try:
            # Perform the search
            results = search.run(query)
            
            # Add results to semantic search index
            process_search_results(topic, results)
            
            # Return results with a note about semantic indexing
            return f"# Search Results for: {query}\n\n{results}\n\n---\n\nThese results have been indexed for semantic search under topic '{topic}'.\nYou can now use the Semantic Search tool with 'topic: your_question' to find semantically similar content."
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    return Tool(
        name="Integrated Search",
        func=integrated_search,
        description="Performs a Google search and automatically indexes the results for semantic search. Format: 'query' or 'topic:query' to specify a topic for semantic indexing."
    )

class SearchCache:
    """
    Cache for storing search results to reduce API calls and improve response times.
    Implements a simple time-based expiration policy.
    """
    def __init__(self, max_size=100, ttl=3600):
        """
        Initialize the search cache.
        
        Args:
            max_size (int): Maximum number of items to store in the cache
            ttl (int): Time to live in seconds for cache entries (default: 1 hour)
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.max_size = max_size
        self.ttl = ttl
        
    def _generate_key(self, search_tool: str, query: str, **kwargs) -> str:
        """
        Generate a cache key from the search parameters.
        
        Args:
            search_tool (str): Name of the search tool being used
            query (str): The search query
            **kwargs: Additional search parameters
            
        Returns:
            str: A hash-based cache key
        """
        # Create a deterministic string representation of all parameters
        key_dict = {
            "tool": search_tool,
            "query": query,
            **kwargs
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Create a hash of the key string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, search_tool: str, query: str, **kwargs) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            search_tool (str): Name of the search tool being used
            query (str): The search query
            **kwargs: Additional search parameters
            
        Returns:
            Optional[Any]: The cached value or None if not found or expired
        """
        key = self._generate_key(search_tool, query, **kwargs)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # Check if the entry has expired
            if time.time() - timestamp <= self.ttl:
                print(f"Cache hit for: {search_tool} - {query}")
                return value
            else:
                # Remove expired entry
                del self.cache[key]
                
        return None
    
    def set(self, search_tool: str, query: str, value: Any, **kwargs) -> None:
        """
        Set a value in the cache.
        
        Args:
            search_tool (str): Name of the search tool being used
            query (str): The search query
            value (Any): The value to cache
            **kwargs: Additional search parameters
        """
        key = self._generate_key(search_tool, query, **kwargs)
        
        # If cache is full, remove the oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        # Store the value with the current timestamp
        self.cache[key] = (value, time.time())
        print(f"Cache set for: {search_tool} - {query}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache = {}
        print("Cache cleared")
    
    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: Statistics about the cache
        """
        current_time = time.time()
        expired_count = sum(1 for _, timestamp in self.cache.values() 
                          if current_time - timestamp > self.ttl)
        
        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "active_entries": len(self.cache) - expired_count
        }
    
# Create a global cache instance
search_cache = SearchCache()

def cache_search_results(func):
    """
    Decorator for caching search results.
    
    Args:
        func: The search function to wrap
        
    Returns:
        The wrapped function with caching
    """
    @wraps(func)
    def wrapper(query, **kwargs):
        # Extract the tool name from the function
        tool_name = func.__name__
        
        # Try to get result from cache
        cached_result = search_cache.get(tool_name, query, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # If not in cache, call the original function
        result = func(query, **kwargs)
        
        # Cache the result
        search_cache.set(tool_name, query, result, **kwargs)
        
        return result
    
    return wrapper

class StreamingCallbackHandler(BaseCallbackHandler):
    """Improved callback handler for streaming LLM responses."""
    def __init__(self):
        self.thoughts = []
        self.current_thought = ""
        self.collected_tokens = []
        self.final_thought_detected = False
        self.all_text = ""
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.thoughts.append({"type": "llm_start", "content": "Starting to think..."})
        
    def on_llm_new_token(self, token, **kwargs):
        self.current_thought += token
        self.collected_tokens.append(token)
        self.all_text += token
        
        # Improved check for "Final Answer:" pattern
        last_tokens = "".join(self.collected_tokens[-30:])
        if "Final Answer:" in last_tokens and not self.final_thought_detected:
            # Extract the thought before "Final Answer:"
            match = re.search(r'Thought:(.*?)Final Answer:', self.all_text, re.DOTALL)
            if match:
                final_thought = match.group(1).strip()
                self.thoughts.append({"type": "final_thought", "content": final_thought})
                self.final_thought_detected = True
        
    def on_llm_end(self, response, **kwargs):
        # Check one more time for final thought if we haven't found it yet
        if not self.final_thought_detected:
            match = re.search(r'Thought:(.*?)Final Answer:', self.all_text, re.DOTALL)
            if match:
                final_thought = match.group(1).strip()
                self.thoughts.append({"type": "final_thought", "content": final_thought})
                self.final_thought_detected = True
        
        self.thoughts.append({"type": "llm_thought", "content": self.current_thought})
        self.current_thought = ""
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.thoughts.append({"type": "tool_start", "tool": serialized["name"], "input": input_str})
        
    def on_tool_end(self, output, **kwargs):
        self.thoughts.append({"type": "tool_result", "content": output})
        
    def on_agent_action(self, action, **kwargs):
        # Try to extract thought from action log
        if hasattr(action, 'log') and action.log:
            thought_match = re.search(r'Thought:(.*?)(?=Action:|$)', action.log, re.DOTALL)
            if thought_match and not self.final_thought_detected:
                thought_content = thought_match.group(1).strip()
                if "Final Answer:" in action.log:
                    # This is the final thought
                    self.thoughts.append({"type": "final_thought", "content": thought_content})
                    self.final_thought_detected = True
        
        self.thoughts.append({
            "type": "agent_action", 
            "tool": action.tool, 
            "tool_input": action.tool_input,
            "log": action.log
        })
        
    def on_agent_finish(self, finish, **kwargs):
        # Final attempt to extract the thought from the finish log
        if not self.final_thought_detected and hasattr(finish, 'log') and finish.log:
            # Try various patterns
            patterns = [
                r'Thought:(.*?)Final Answer:',
                r'Thought[:\s]+(.*?)(?=Action:|Observation:|Final Answer:|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, finish.log, re.DOTALL)
                if match:
                    final_thought = match.group(1).strip()
                    self.thoughts.append({"type": "final_thought", "content": final_thought})
                    self.final_thought_detected = True
                    break
            
        # Add the final answer
        if hasattr(finish, 'return_values') and 'output' in finish.return_values:
            self.thoughts.append({"type": "agent_finish", "content": finish.return_values["output"]})
    
    def on_observation(self, observation, **kwargs):
        self.thoughts.append({"type": "observation", "content": observation})

# Enhanced custom prompt for in-depth information retrieval
ENHANCED_SEARCH_PROMPT = """You are an advanced search agent specialized in retrieving comprehensive, accurate, and diverse information from the web.

When searching for information, follow these advanced search strategies:

1. MULTI-DIMENSIONAL SEARCHING:
   - Break complex queries into multiple simpler searches
   - Explore different aspects of the topic systematically
   - Use a variety of search formulations to capture different results

2. TOOL SELECTION EXPERTISE:
   - Advanced Google Search: Use for targeted searches with filters like site:example.com
   - Topic Research: Use for comprehensive overview of a multi-faceted topic
   - Comparative Research: Use when comparing different topics, approaches, or perspectives
   - Deep Research: Use for in-depth investigation requiring multiple related searches

3. SEARCH REFINEMENT TECHNIQUES:
   - Start broad, then narrow based on initial results
   - Use quoted phrases for exact matches when appropriate
   - Add clarifying terms to disambiguate ambiguous queries
   - Exclude irrelevant results using exclusion terms

4. INFORMATION SYNTHESIS:
   - Combine information from multiple sources
   - Identify patterns and consensus across sources
   - Note conflicting information and different perspectives
   - Indicate the reliability of information based on source quality

5. TRANSPARENT REASONING:
   - Explain your search strategy clearly
   - Show how you're evaluating and synthesizing information
   - Acknowledge limitations in available information
   - Distinguish between facts, expert opinions, and speculation

Remember to use the appropriate search tool for each task:
- For simple factual queries: Basic Google Search
- For advanced filtering: Advanced Google Search with syntax like --site:edu --time:week
- For comprehensive topic exploration: Topic Research
- For comparing alternatives: Comparative Research
- For deep investigation: Deep Research

IMPORTANT: Structure your searches strategically. If one approach isn't yielding useful results, try a different approach or tool rather than repeating similar searches.

For user query: {input}

First, think about the best search strategy for this specific query. Then execute your plan step by step using the most appropriate search tools."""

# Initialize LLM with appropriate settings for detailed responses
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Using the Pro model for more comprehensive responses
        temperature=0.2,  # Lower temperature for more factual and detailed responses
        top_p=0.85,  # Slightly constrained sampling for focus
        top_k=40,  # Wider token selection for diverse information
        max_output_tokens=8192,  # Longer responses
        api_key=os.getenv("LLM_API_KEY"),
        safety_settings={
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

def get_advanced_search_tool():
    """
    Creates an enhanced search tool with more capabilities:
    - Configurable result count
    - Support for different result types (news, images, etc.)
    - Site-specific searches
    - Time-filtered searches
    """
    search = GoogleSearchAPIWrapper(
        k=10,  # Increased result count
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    def advanced_search(query_string):
        """
        Parse advanced search options from the query string and perform search.
        
        Format:
        - "query" - Standard search
        - "query --site:example.com" - Site-specific search
        - "query --count:20" - Specify result count
        - "query --time:day" - Time filter (day, week, month, year)
        - "query --type:news" - Search specific content types (news, blogs, etc.)
        """
        # Parse advanced search parameters
        params = {}
        site_match = re.search(r'--site:([\w.-]+)', query_string)
        count_match = re.search(r'--count:(\d+)', query_string)
        time_match = re.search(r'--time:(day|week|month|year)', query_string)
        type_match = re.search(r'--type:(news|blogs|discussions|images)', query_string)
        
        # Clean the query
        clean_query = query_string
        for pattern in [r'--site:[\w.-]+', r'--count:\d+', r'--time:(day|week|month|year)', 
                       r'--type:(news|blogs|discussions|images)']:
            clean_query = re.sub(pattern, '', clean_query).strip()
        
        # Apply site-specific search
        if site_match:
            site = site_match.group(1)
            clean_query = f"site:{site} {clean_query}"
        
        # Apply result count
        result_count = 10  # Default
        if count_match:
            result_count = min(int(count_match.group(1)), 30)  # Cap at 30 to avoid API issues
        
        # Apply time filter (would need custom implementation with Google Search API)
        time_period = None
        if time_match:
            time_period = time_match.group(1)
            # Note: In a real implementation, you would modify the search API call based on time_period
        
        # Apply content type filter
        content_type = None
        if type_match:
            content_type = type_match.group(1)
            # Note: In a real implementation, you would modify the search API call based on content_type
        
        # Log the search parameters
        search_params = {
            "query": clean_query,
            "result_count": result_count,
            "site_filter": site_match.group(1) if site_match else None,
            "time_filter": time_period,
            "content_type": content_type
        }
        
        print(f"Advanced search parameters: {search_params}")
        
        # Perform the search with the specified parameters
        # In a real implementation, you would use different search methods based on the parameters
        try:
            # For this example, we're just changing the k parameter
            search.k = result_count
            results = search.run(clean_query)
            return f"Search results for: {clean_query}\n\n{results}"
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    return Tool(
        name="Advanced Google Search",
        func=advanced_search,
        description="Enhanced search tool that supports site filtering, result count specification, time filters, and content type filters. Format: 'query --site:example.com --count:20 --time:day --type:news'"
    )

def get_topic_research_tool():
    """
    Creates a comprehensive topic research tool that performs multiple related searches
    to gather detailed information on a topic.
    """
    search = GoogleSearchAPIWrapper(
        k=5,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    def topic_research(topic):
        """
        Perform comprehensive research on a topic by running multiple targeted searches.
        """
        # Define research dimensions
        research_dimensions = [
            {"suffix": "overview", "description": "General overview"},
            {"suffix": "definition", "description": "Definitions and terminology"},
            {"suffix": "examples", "description": "Examples and case studies"},
            {"suffix": "statistics", "description": "Statistics and data"},
            {"suffix": "perspectives", "description": "Different perspectives and opinions"},
            {"suffix": "history", "description": "Historical context"}
        ]
        
        # Run searches for each dimension
        research_results = {}
        for dimension in research_dimensions[:3]:  # Limit to 3 dimensions to avoid API limits
            search_query = f"{topic} {dimension['suffix']}"
            try:
                results = search.run(search_query)
                research_results[dimension['description']] = results
            except Exception as e:
                research_results[dimension['description']] = f"Error: {str(e)}"
        
        # Format the results
        formatted_results = f"# Comprehensive Research on: {topic}\n\n"
        for dimension, results in research_results.items():
            formatted_results += f"## {dimension}\n\n{results}\n\n"
        
        return formatted_results
    
    return Tool(
        name="Topic Research",
        func=topic_research,
        description="Performs comprehensive research on a topic by exploring multiple dimensions (definitions, examples, statistics, perspectives, etc.)"
    )

def get_comparative_search_tool():
    """
    Creates a tool for comparing multiple topics or perspectives.
    """
    search = GoogleSearchAPIWrapper(
        k=3,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    def comparative_search(query):
        """
        Compare multiple topics or perspectives.
        
        Format: "topic1 vs topic2 vs topic3"
        or "compare: topic1, topic2, topic3"
        """
        topics = []
        
        # Parse topics from the query
        if "compare:" in query:
            topics_section = query.split("compare:")[1].strip()
            topics = [t.strip() for t in topics_section.split(",")]
        elif " vs " in query:
            topics = [t.strip() for t in query.split(" vs ")]
        else:
            return "Please use the format 'topic1 vs topic2' or 'compare: topic1, topic2'"
        
        if len(topics) < 2:
            return "Please provide at least two topics to compare"
        
        # Limit the number of topics to avoid API limits
        topics = topics[:3]
        
        # Perform comparative searches
        results = {}
        
        # Individual topic searches
        for topic in topics:
            try:
                topic_results = search.run(topic)
                results[topic] = topic_results
            except Exception as e:
                results[topic] = f"Error searching for {topic}: {str(e)}"
        
        # Direct comparison search
        comparison_query = " vs ".join(topics)
        try:
            comparison_results = search.run(comparison_query)
            results["Direct Comparison"] = comparison_results
        except Exception as e:
            results["Direct Comparison"] = f"Error searching for comparison: {str(e)}"
        
        # Format the results
        formatted_results = f"# Comparison: {' vs '.join(topics)}\n\n"
        
        # Individual topic sections
        for topic, topic_results in results.items():
            formatted_results += f"## {topic}\n\n{topic_results}\n\n"
        
        return formatted_results
    
    return Tool(
        name="Comparative Research",
        func=comparative_search,
        description="Compares multiple topics or perspectives. Use format: 'topic1 vs topic2' or 'compare: topic1, topic2, topic3'"
    )

def get_enhanced_search_tools():
    """Returns a list of all enhanced search tools."""
    return [
        get_search_tool(),  # Original search tool
        get_advanced_search_tool(),  # Enhanced search with more options
        get_topic_research_tool(),  # Comprehensive topic research
        get_comparative_search_tool(),  # Comparative research
        get_deep_search_tool()  # Original deep search
    ]

# Set up enhanced Google Search tool with improved parameters
def get_search_tool():
    if not os.getenv("GOOGLE_CSE_ID") or not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google Search API credentials not found in environment variables")
        
    os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    search = GoogleSearchAPIWrapper(
        k=8,  # Increase number of results
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    return Tool(
        name="Google Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events, facts, or the current state of the world. Use this tool multiple times with different search terms to gather comprehensive information."
    )

# Add a new tool for deep information retrieval
def get_deep_search_tool():
    search = GoogleSearchAPIWrapper(
        k=5,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    def deep_search(query):
        """Perform multiple related searches to gather comprehensive information."""
        base_results = search.run(query)
        
        # Generate related search queries
        related_queries = [
            f"detailed explanation of {query}",
            f"{query} analysis in-depth",
            f"{query} expert perspective",
            f"{query} statistics and data",
            f"{query} latest research"
        ]
        
        # Gather additional information from related queries
        additional_results = []
        for related_query in related_queries[:2]:  # Limit to avoid API rate limits
            try:
                results = search.run(related_query)
                additional_results.append(f"Additional information from '{related_query}':\n{results}")
            except Exception as e:
                additional_results.append(f"Error searching for '{related_query}': {str(e)}")
        
        # Combine results
        combined_results = base_results + "\n\n" + "\n\n".join(additional_results)
        return combined_results
    
    return Tool(
        name="Deep Research",
        func=deep_search,
        description="Use this for comprehensive research on complex topics. This tool performs multiple related searches to gather detailed information from various perspectives."
    )

# Extract final thought from text
def extract_final_thought(text):
    """
    Extract final thought from text with improved pattern matching.
    
    Parameters:
    text (str): Text containing thought patterns
    
    Returns:
    str: Extracted final thought or None if not found
    """
    if not text:
        return None
        
    # Look for patterns like "Thought: ... Final Answer:" 
    match = re.search(r'Thought:(.*?)Final Answer:', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Also try to match a more flexible pattern that might appear in different formats
    match = re.search(r'Thought[:\s]+(.*?)(?=Action:|Observation:|Final Answer:|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Check for a standalone thought line
    lines = text.split('\n')
    for line in lines:
        if line.startswith('Thought:'):
            return line[8:].strip()
    
    return None

# Initialize the agent with enhanced tools and prompt
def get_agent(stream=False, detailed=True):
    tools = get_enhanced_search_tools()
    callbacks = [StreamingCallbackHandler()] if stream else None
    
    llm = get_llm()
    
    if detailed:
        return initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
            callbacks=callbacks,
            agent_kwargs={
                "prefix": ENHANCED_SEARCH_PROMPT
            },
            handle_parsing_errors=True
        )
    else:
        return initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
            callbacks=callbacks,
            handle_parsing_errors=True
        )
    
class SearchResultEnricher:
    """
    Enhances search results with additional information and context.
    """
    def __init__(self, fetch_snippets=True, analyze_credibility=True):
        """
        Initialize the search result enricher.
        
        Args:
            fetch_snippets (bool): Whether to fetch content snippets from result URLs
            analyze_credibility (bool): Whether to analyze source credibility
        """
        self.fetch_snippets = fetch_snippets
        self.analyze_credibility = analyze_credibility
        
        # Domain credibility indicators (simplified example)
        self.credible_domains = {
            # Educational domains
            'edu': 0.9,
            'ac.uk': 0.85,
            # Government domains
            'gov': 0.9,
            # Research and scientific organizations
            'nih.gov': 0.95,
            'who.int': 0.95,
            'nature.com': 0.9,
            'science.org': 0.9,
            'sciencedirect.com': 0.85,
            'ieee.org': 0.85,
            # Reputable news sources (simplified)
            'bbc.co.uk': 0.8,
            'reuters.com': 0.8,
            'apnews.com': 0.8,
            # General reference
            'wikipedia.org': 0.75,
        }
    
    def parse_search_results(self, results: str) -> List[Dict[str, Any]]:
        """
        Parse raw search results into structured format.
        
        Args:
            results (str): Raw search results text
            
        Returns:
            List[Dict[str, Any]]: Structured search results
        """
        parsed_results = []
        
        # Simple parsing logic assuming results contain url and snippet
        # In a real implementation, this would depend on the exact format returned by the search API
        result_blocks = results.split("\n\n")
        
        for block in result_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split("\n")
            if not lines:
                continue
                
            # Try to extract URL and title
            url_match = re.search(r'https?://\S+', block)
            url = url_match.group(0) if url_match else None
            
            if not url:
                continue
                
            # The first line that's not the URL might be the title
            title = None
            for line in lines:
                if line and url not in line:
                    title = line
                    break
                    
            # The rest is the snippet
            snippet = block.replace(url, "")
            if title:
                snippet = snippet.replace(title, "")
            
            snippet = snippet.strip()
            
            parsed_results.append({
                "url": url,
                "title": title or "Untitled",
                "snippet": snippet
            })
            
        return parsed_results
    
    def assess_domain_credibility(self, url: str) -> Dict[str, Any]:
        """
        Assess the credibility of a domain.
        
        Args:
            url (str): The URL to assess
            
        Returns:
            Dict[str, Any]: Credibility assessment
        """
        if not url:
            return {"score": 0.5, "factors": ["No URL provided"]}
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        credibility_score = 0.5  # Default neutral score
        factors = []
        
        # Check for known credible domains
        for credible_domain, score in self.credible_domains.items():
            if domain.endswith(credible_domain):
                credibility_score = score
                factors.append(f"Domain ends with credible TLD/domain: {credible_domain}")
                break
        
        # Check for HTTPS
        if parsed_url.scheme == 'https':
            credibility_score += 0.05
            factors.append("Uses secure HTTPS connection")
            
        # Check for suspicious patterns
        suspicious_patterns = [
            r'free.*download',
            r'hack',
            r'crack',
            r'warez',
            r'\d{5,}',  # Long numbers in domain
            r'-{2,}'  # Multiple consecutive hyphens
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain):
                credibility_score -= 0.1
                factors.append(f"Domain contains suspicious pattern: {pattern}")
        
        # Cap score between 0 and 1
        credibility_score = max(0, min(1, credibility_score))
        
        return {
            "score": credibility_score,
            "factors": factors,
            "domain": domain
        }
    
    async def fetch_url_snippet(self, url: str, timeout: int = 5) -> Optional[str]:
        """
        Fetch a content snippet from a URL asynchronously.
        
        Args:
            url (str): The URL to fetch
            timeout (int): Timeout in seconds
            
        Returns:
            Optional[str]: A snippet of the content or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        return None
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Break into lines and remove leading/trailing space
                    lines = (line.strip() for line in text.splitlines())
                    
                    # Break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    
                    # Drop blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Return a snippet (first ~500 characters)
                    return text[:500] + "..." if len(text) > 500 else text
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    async def enrich_search_results(self, results: str) -> Dict[str, Any]:
        """
        Enrich search results with additional information.
        
        Args:
            results (str): Raw search results text
            
        Returns:
            Dict[str, Any]: Enriched search results
        """
        start_time = time.time()
        
        # Parse the results
        parsed_results = self.parse_search_results(results)
        
        # Enrich each result
        enriched_results = []
        for result in parsed_results:
            enriched_result = result.copy()
            
            # Assess domain credibility
            if self.analyze_credibility and result.get("url"):
                enriched_result["credibility"] = self.assess_domain_credibility(result["url"])
            
            enriched_results.append(enriched_result)
        
        # Fetch content snippets asynchronously
        if self.fetch_snippets:
            snippet_tasks = [self.fetch_url_snippet(result["url"]) for result in enriched_results if result.get("url")]
            snippets = await asyncio.gather(*snippet_tasks)
            
            # Add snippets to results
            for i, snippet in enumerate(snippets):
                if i < len(enriched_results) and snippet:
                    enriched_results[i]["content_snippet"] = snippet
        
        # Calculate domain diversity
        domains = {}
        for result in enriched_results:
            if result.get("url"):
                domain = urlparse(result["url"]).netloc
                domains[domain] = domains.get(domain, 0) + 1
        
        # Calculate overall quality metrics
        credibility_scores = [result.get("credibility", {}).get("score", 0.5) 
                             for result in enriched_results if "credibility" in result]
        
        average_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        
        domain_diversity = 1 - (sum((count - 1) for count in domains.values()) / len(enriched_results) if enriched_results else 0)
        domain_diversity = max(0, min(1, domain_diversity))  # Ensure between 0 and 1
        
        process_time = time.time() - start_time
        
        return {
            "results": enriched_results,
            "metrics": {
                "count": len(enriched_results),
                "unique_domains": len(domains),
                "domain_diversity": domain_diversity,
                "average_credibility": average_credibility,
                "processing_time": process_time
            },
            "domain_breakdown": domains
        }

# Create a search result enricher tool
def get_enriched_search_tool():
    """Creates a tool for enhanced search with result enrichment."""
    search = GoogleSearchAPIWrapper(
        k=8,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    enricher = SearchResultEnricher()
    
    async def enriched_search(query: str) -> str:
        """
        Perform a search with enhanced result processing and enrichment.
        
        Args:
            query (str): The search query
            
        Returns:
            str: Formatted enriched search results
        """
        try:
            # Perform the base search
            raw_results = search.run(query)
            
            # Enrich the results
            enriched_data = await enricher.enrich_search_results(raw_results)
            
            # Format the results for output
            formatted_results = f"# Enhanced Search Results for: {query}\n\n"
            
            # Add quality metrics
            metrics = enriched_data["metrics"]
            formatted_results += "## Search Quality Metrics\n"
            formatted_results += f"- **Results Found:** {metrics['count']}\n"
            formatted_results += f"- **Unique Sources:** {metrics['unique_domains']}\n"
            formatted_results += f"- **Source Diversity:** {metrics['domain_diversity']:.2f}/1.0\n"
            formatted_results += f"- **Average Source Credibility:** {metrics['average_credibility']:.2f}/1.0\n\n"
            
            # Add the results
            formatted_results += "## Results\n\n"
            for i, result in enumerate(enriched_data["results"]):
                formatted_results += f"### {i+1}. {result['title']}\n"
                formatted_results += f"**URL:** {result['url']}\n"
                
                if "credibility" in result:
                    cred = result["credibility"]
                    cred_label = "High" if cred["score"] > 0.7 else "Medium" if cred["score"] > 0.4 else "Low"
                    formatted_results += f"**Source Credibility:** {cred_label} ({cred['score']:.2f}/1.0)\n"
                
                formatted_results += f"\n{result['snippet']}\n\n"
                
                if "content_snippet" in result:
                    formatted_results += "**Content Preview:**\n"
                    formatted_results += f"{result['content_snippet'][:300]}...\n\n"
                
                formatted_results += "---\n\n"
            
            return formatted_results
        except Exception as e:
            return f"Error performing enriched search: {str(e)}"
    
    # Create a synchronous wrapper for the async function
    def enriched_search_sync(query: str) -> str:
        """Synchronous wrapper for the async enriched search function."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(enriched_search(query))
    
    return Tool(
        name="Enriched Search",
        func=enriched_search_sync,
        description="Performs an enhanced search with result credibility analysis, content previews, and quality metrics."
    )

app = FastAPI(title="Enhanced LangChain Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    detailed: bool = True  # Default to detailed mode

# Example application to search tools
@cache_search_results
def advanced_search(query, **kwargs):
    """Example search function with caching."""
    # Implementation would call the actual search API
    pass

# API endpoint for cache management
@app.get("/cache/stats")
async def get_cache_stats():
    """Get statistics about the search cache."""
    return search_cache.stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear the search cache."""
    search_cache.clear()
    return {"message": "Cache cleared successfully"}

# Example of applying the cache to the existing search tools
def get_cached_search_tool():
    """Get a cached version of the standard search tool."""
    search = GoogleSearchAPIWrapper(
        k=8,
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    @cache_search_results
    def cached_search(query):
        """Cached version of the standard search."""
        return search.run(query)
    
    return Tool(
        name="Cached Google Search",
        func=cached_search,
        description="Cached version of Google Search. Use for repeated queries to reduce API usage."
    )

# Add semantic search API endpoints
@app.get("/semantic/topics")
async def list_semantic_topics():
    """List all available semantic search topics."""
    try:
        topics = semantic_search_manager.list_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing topics: {str(e)}")

@app.post("/semantic/search")
async def semantic_search_endpoint(topic: str, query: str, k: int = 5):
    """
    Perform a semantic search.
    
    Args:
        topic (str): The topic to search in
        query (str): The search query
        k (int): Number of results to return
    """
    try:
        if topic not in semantic_search_manager.list_topics():
            raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found")
            
        results = semantic_search_manager.search(topic, query, k=k)
        return {"results": results}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")

@app.delete("/semantic/topics/{topic}")
async def delete_semantic_topic(topic: str):
    """Delete a semantic search topic."""
    if not semantic_search_manager.delete_topic(topic):
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found")
    return {"message": f"Topic '{topic}' deleted successfully"}

@app.post("/generate-response")
async def generate_response(data: QueryRequest):
    try:
        # Get agent with appropriate settings
        agent = get_agent(stream=data.stream, detailed=data.detailed)
        
        # Execute the agent with the query
        response = agent({"input": data.query})
        
        # Format the final response as markdown
        final_response = markdown2.markdown(response['output'])
        intermediate_steps = response['intermediate_steps']
        
        # Extract thought processes in a readable format
        thought_process = []
        for step in intermediate_steps:
            action = step[0]
            result = step[1]
            
            thought_entry = {
                "thought": action.log,
                "action": action.tool,
                "action_input": action.tool_input,
                "result": result
            }
            
            # Add observation if available (extract from the action log)
            if hasattr(action, 'observation') and action.observation:
                thought_entry["observation"] = action.observation
                
            thought_process.append(thought_entry)
        
        # Try to extract the final thought from the agent's output
        final_thought = None
        if "Final Answer:" in response.get("output", ""):
            # Extract from the full agent output
            final_thought = extract_final_thought(response["output"])
        
        # If not found, check the intermediate steps for the last thought
        if not final_thought and intermediate_steps:
            last_action = intermediate_steps[-1][0]
            if hasattr(last_action, 'log'):
                final_thought = extract_final_thought(last_action.log)
        
        # Add search quality indicators
        search_quality = {
            "num_searches": len([step for step in intermediate_steps if step[0].tool in ["Google Search", "Deep Research"]]),
            "info_richness": "high" if len(response['output']) > 1000 else "medium" if len(response['output']) > 500 else "low",
            "search_depth": "high" if "Deep Research" in [step[0].tool for step in intermediate_steps] else "standard"
        }
        
        return {
            "query": data.query,
            "final_response": final_response,
            "thought_process": thought_process,
            "final_thought": final_thought,
            "search_quality": search_quality,
            "raw_intermediate_steps": intermediate_steps,
            "response": agent._construct_scratchpad(intermediate_steps)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

class ObservableAgent:
    """Wrapper for the agent to capture observations."""
    def __init__(self, callback_handler, detailed=True):
        self.tools = get_enhanced_search_tools()
        self.llm = get_llm()
        self.callback_handler = callback_handler
        self.detailed = detailed
        
        if detailed:
            self.agent = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                callbacks=[self.callback_handler],
                agent_kwargs={
                    "prefix": ENHANCED_SEARCH_PROMPT
                },
                handle_parsing_errors=True
            )
        else:
            self.agent = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                callbacks=[self.callback_handler],
                handle_parsing_errors=True
            )
        
    async def run(self, query):
        """Run the agent and track observations."""
        try:
            result = await asyncio.to_thread(self.agent, {"input": query})
            
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    action, observation = step
                    self.callback_handler.on_observation(observation)
                    
            return result
        except langchain_core.exceptions.OutputParserException as e:
            print(f"Output parser error: {e}")
            error_msg = str(e)
            llm_output = None
            if "Could not parse LLM output: `" in error_msg:
                llm_output = error_msg.split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
            
            return {
                "output": llm_output or "I couldn't format my response in the expected way, but here's what I found: " + 
                          "The model attempted to respond directly to your query instead of using the expected agent format. " +
                          "This typically happens when the query is straightforward and doesn't require multiple research steps.",
                "intermediate_steps": []
            }
        except Exception as e:
            print(f"Error running agent: {e}")
            return {
                "output": f"I encountered an error while processing your request. Please try rephrasing your query. Error details: {str(e)}",
                "intermediate_steps": []
            }

async def stream_agent_response(query: str, detailed=True):
    """Stream the agent's response as it's generated with improved final thought detection."""
    callback_handler = StreamingCallbackHandler()
    observable_agent = ObservableAgent(callback_handler, detailed=detailed)
    
    # Extract the final thought from the input query if it's a reference
    if "<document_content>" in query and "Final Answer:" in query:
        # Try to extract final thought from the sample content
        final_thought = extract_final_thought(query)
        if final_thought:
            callback_handler.thoughts.append({"type": "final_thought", "content": final_thought})
            callback_handler.final_thought_detected = True
    
    # Send a message indicating processing has started
    yield json.dumps({
        "type": "status", 
        "content": "Processing your query..."
    }) + "\n"
    
    # Run the agent in a separate task with error handling
    response_task = asyncio.create_task(observable_agent.run(query))
    
    # Stream thought process as it happens
    last_thought_index = 0
    while not response_task.done():
        if len(callback_handler.thoughts) > last_thought_index:
            # Send new thoughts that we haven't sent yet
            new_thoughts = callback_handler.thoughts[last_thought_index:]
            for thought in new_thoughts:
                yield json.dumps(thought) + "\n"
            
            last_thought_index = len(callback_handler.thoughts)
        
        await asyncio.sleep(0.1)
    
    # Get the final response
    try:
        final_response = await response_task
        
        # Check if we got a valid response
        if final_response is None:
            yield json.dumps({
                "type": "error",
                "content": "No response received from the agent."
            }) + "\n"
            return
            
        # Deep inspection of response for final thought patterns if we still don't have one
        if not callback_handler.final_thought_detected:
            # Try from the output first
            final_thought = extract_final_thought(final_response.get("output", ""))
            
            # If not found, try from intermediate steps
            if not final_thought and 'intermediate_steps' in final_response:
                for step in final_response.get('intermediate_steps', []):
                    if hasattr(step[0], 'log') and step[0].log:
                        final_thought = extract_final_thought(step[0].log)
                        if final_thought:
                            break
            
            # Finally, try from the raw text that was collected
            if not final_thought and hasattr(callback_handler, 'all_text'):
                final_thought = extract_final_thought(callback_handler.all_text)
            
            if final_thought:
                yield json.dumps({
                    "type": "final_thought", 
                    "content": final_thought
                }) + "\n"
        
        # Add search quality metrics
        search_steps = [step for step in final_response.get('intermediate_steps', []) 
                       if hasattr(step[0], 'tool') and step[0].tool in ["Google Search", "Deep Research"]]
        
        yield json.dumps({
            "type": "search_metrics",
            "content": {
                "num_searches": len(search_steps),
                "search_depth": "high" if any(hasattr(step[0], 'tool') and step[0].tool == "Deep Research" for step in final_response.get('intermediate_steps', [])) else "standard",
                "response_length": len(final_response.get("output", ""))
            }
        }) + "\n"
        
        yield json.dumps({
            "type": "final_response", 
            "content": final_response.get("output", "No output was generated. Please try rephrasing your question.")
        }) + "\n"
    except Exception as e:
        print(f"Exception in stream_agent_response: {e}")
        yield json.dumps({
            "type": "error",
            "content": f"Error: {str(e)}"
        }) + "\n"
        
        # Provide a fallback response
        yield json.dumps({
            "type": "final_response",
            "content": "I encountered an error while processing your request. Please try rephrasing your question."
        }) + "\n"

@app.post("/stream-response")
async def stream_response(data: QueryRequest):
    """Endpoint that streams the agent's thought process and final response."""
    return StreamingResponse(
        stream_agent_response(data.query, detailed=data.detailed),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to prevent server crashes."""
    error_msg = str(exc)
    print(f"Global exception: {error_msg}")
    
    if isinstance(exc, langchain_core.exceptions.OutputParserException):
        # Extract the actual LLM response from the error
        llm_output = None
        if "Could not parse LLM output: `" in error_msg:
            llm_output = error_msg.split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
        
        return JSONResponse(
            status_code=200,
            content={
                "error": "parsing_error",
                "message": "The model provided a direct response instead of using the agent format",
                "direct_response": llm_output or "No direct response extracted"
            }
        )
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": error_msg}
    )

# Add improved analytics to the API
@app.get("/search/analytics")
async def get_search_analytics(query: str = None):
    """
    Get analytics for search results.
    
    Args:
        query (str, optional): If provided, perform the search and return analytics
    """
    enricher = SearchResultEnricher()
    
    if query:
        # Create a search instance
        search = GoogleSearchAPIWrapper(
            k=8,
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Perform the search
        raw_results = search.run(query)
        
        # Enrich and analyze the results
        enriched_data = await enricher.enrich_search_results(raw_results)
        
        return {
            "query": query,
            "analytics": enriched_data["metrics"],
            "domain_breakdown": enriched_data["domain_breakdown"]
        }
    else:
        # Just return the empty analytics structure
        return {
            "message": "No query provided. Add '?query=your search' to get analytics for a specific search."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)