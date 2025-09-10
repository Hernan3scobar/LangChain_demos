#

# %%
from dotenv import load_dotenv

load_dotenv("/home/hernan/langchain-course/.env", override=True)


from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_ollama import OllamaLLM
from langchain_tavily import TavilySearch
from langsmith import traceable

tools = [TavilySearch()]

llm = OllamaLLM(model="gemma2:2b", base_url="http://localhost:11434")

react_prom_v1 = hub.pull("kenwu/gemma-json-react")
react_promt_v2 = hub.pull("hwchase17/react")







agent = create_react_agent(llm, tools, react_promt_v2)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor
promt = "Search 3 AI engineer positions in Chile using linkedint and list their details and salaries?"


@traceable
def react_agent():
    result = chain.invoke(input={"input": promt})
    print(result)


if __name__ == "__main__":
    react_agent()


# %%
