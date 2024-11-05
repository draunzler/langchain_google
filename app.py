from flask import Flask, render_template, request
import os
from dotenv import load_dotenv  # Import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool, AgentType, initialize_agent
import markdown2

# Load environment variables from .env file
load_dotenv()

# Initialize LLM with the API key from the environment variable
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("LLM_API_KEY"),  # Use os.getenv to get the API key
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

# Set up environment variables for Google API
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")  # Use os.getenv to get the CSE ID
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Use os.getenv to get the API key

# Initialize Google Search and tools
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events or the current state of the world"
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True
)

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    final_response = None
    intermediate_steps = []

    if request.method == 'POST':
        query = request.form.get('query')

        response = agent({"input": query})
        
        final_response = markdown2.markdown(response['output'])
        intermediate_steps = response['intermediate_steps']
    
    return render_template('index.html', query=query, final_response=final_response, intermediate_steps=intermediate_steps)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
