import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tools import AnalysisTool, SearchTool, ComparisonTool

# Initialize tools
search_tool = SearchTool()
comparison_tool = ComparisonTool()
analysis_tool = AnalysisTool()

tools = [search_tool, comparison_tool, analysis_tool]

# Define the prompt template
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt_template = PromptTemplate.from_template(template)

# Initialize the ReAct agent
agent = create_react_agent(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
    tools=tools,
    prompt=prompt_template
)

# Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Streamlit Interface
st.title("ReAct Agent Interface")

st.subheader("Enter your query")
query = st.text_input("Ask a question about camera quality, device comparisons, etc.")
submit_button = st.button("Submit")

if submit_button and query:
    st.subheader("Results")
    try:
        response = agent_executor.invoke({"input": query})
        st.success(response['output'])
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Optional: Display reasoning steps if verbose output is desired
    if st.checkbox("Show reasoning steps"):
        st.write(response.get('reasoning', 'No reasoning steps available.'))