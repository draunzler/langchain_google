import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain.agents import initialize_agent, Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Union
import requests

load_dotenv()

app = FastAPI()

# Google Custom Search API endpoint and API key
GOOGLE_API_KEY = "your_google_api_key"
CSE_ID = "your_cse_id"

@tool
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

# Initialize the LangChain agent using Gemini-2.0-Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tools = [get_advanced_search_tool]
agent = initialize_agent(tools, llm, verbose=True)

def _construct_scratchpad(
    self, intermediate_steps: List[Tuple[AgentAction, str]]
) -> str:
    agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
    if not isinstance(agent_scratchpad, str):
        raise ValueError("agent_scratchpad should be of type string.")
    if agent_scratchpad:
        return (
            f"This was your previous work "
            f"(but I haven't seen any of it! I only see what "
            f"you return as final answer):\n{agent_scratchpad}"
        )
    else:
        return agent_scratchpad

@app.get("/search")
def search_google(query: str = Query(...)):
    try:
        intermediate_steps: List[Tuple[Union[AgentAction, AgentFinish], str]] = []
        response = agent.run(query, callbacks=[intermediate_steps.append])
        final_thought_process = _construct_scratchpad(intermediate_steps)
        return {"final_thought": final_thought_process, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))