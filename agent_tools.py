from langchain.agents import Tool
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import Tool
import os
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from typing import List, Dict
import re

load_dotenv()

class SearchTool(Tool):
    def __init__(self, api_key=None):
        super().__init__(name="search", description="Performs a web search using SerpAPI.", func=self._run)

    def _run(self, query: str) -> str:
        """Run the search query using SerpAPI and return formatted results."""
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERP_API_KEY"),
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Format the results
        formatted_results = self.format_results(results)
        
        # Join results into a single string for easier display in the ReAct agent
        formatted_output = "\n".join(
            [f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\n" for item in formatted_results]
        )
        
        return formatted_output

    def format_results(self, results):
        """Format search results to return only relevant information."""
        formatted_results = []
        
        if "organic_results" in results:
            for result in results["organic_results"]:
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                
                # Each result is a dictionary with title, link, and snippet for easy access by the ReAct agent
                formatted_results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })
        else:
            formatted_results.append({"error": "No results found"})

        return formatted_results

class ComparisonTool(Tool):
    def __init__(self):
        super().__init__(
            name="ComparisonTool",
            description="Compares items based on a specified category",
            func=self._run  # Pass _run as the function to execute
        )
        
    def _run(self, tool_input: str):
        # Parse items and category from the input string
        match = re.search(r'items: \[(.*?)\], category: "|\'(.*?)"|\'', tool_input)
        if match:
            items = [item.strip() for item in match.group(1).split(",")]
            category = match.group(2)
        else:
            return f"Error: Input format is incorrect. Expected format is 'items: [item1, item2, ...], category: \"category_name\"'. Received: {tool_input}"
        
        # Error handling for invalid inputs
        if not items or len(items) < 2:
            return "Error: Provide at least two items for comparison."
        if not category:
            return "Error: Please specify a category for comparison."
        
        # Format the items and apply the prompt template
        prompt_template = PromptTemplate(
            input_variables=["items", "category"],
            template="Compare the following items based on {category}. Provide a summary highlighting key differences and similarities:\n\n{items}"
        )
        formatted_items = "\n".join([f"- {item}" for item in items])
        prompt = prompt_template.format(items=formatted_items, category=category)
        
        # Comparison logic (simulated response for this example)
        result = self.compare_items(prompt)
        
        return result

    def compare_items(self, prompt: str) -> str:
        # This function simulates a response from a model using the prompt.
        # Replace with actual model call in a production setting.
        response = f"Comparison based on {prompt.split('based on')[1].split('.')[0]}:\n"
        response += "Key points:\n1. Item A has ...\n2. Item B differs by ...\n3. Similarities include ..."
        return response

class AnalysisTool(Tool):
    def __init__(self, api_key=None):
        super().__init__(name="analyze", description="Analyzes and summarizes search or comparison results.", func=self._run)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def _run(self, content):
        # Accepts plain text summaries or lists of snippets
        if not content or len(content) == 0:
            return "Error: No content provided for analysis."

        # If content is a list of dicts, convert to a formatted string
        if isinstance(content, list) and all(isinstance(item, dict) for item in content):
            content = self.format_content(content)
        elif isinstance(content, str):
            # If content is already a string, use it directly
            pass
        else:
            return "Error: Unsupported content format. Please provide a list of dictionaries or a summary string."

        prompt = self.create_prompt(content)

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error in generating analysis: {str(e)}"

    def format_content(self, content):
        """Formats list of dictionaries for analysis."""
        formatted_content = ""
        for item in content:
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet available")
            formatted_content += f"Title: {title}\nSnippet: {snippet}\n\n"
        return formatted_content.strip()

    def create_prompt(self, content):
        """Creates a structured prompt for analysis."""
        prompt = (
            "Analyze the following information and provide a concise summary of key points:\n\n"
            f"{content}\n\nSummary:"
        )
        return prompt