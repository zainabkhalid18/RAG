from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

print(f"langgchain-openai version: {ChatOpenAI.__version__}")
def main():
    llm = ChatOpenAI(model_name="gpt-4o" , temperature=0)
    print("Hello from rag!")


if __name__ == "__main__":
    main()
