from pinecone import Pinecone, ServerlessSpec
from disraptors.rag.utils import load_config
secrets = load_config("secret.config")
pinecone = Pinecone(api_key=secrets["pinecone"]["api_key"])
index = pinecone.Index("disraptors-w")
print(index.describe_index_stats())


from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

sonnet = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=secrets["claude"]["api_key"], temperature=0)
gpt4 = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=secrets["openai"]["api_key"], temperature=0)

print(sonnet.invoke("what is quantum physics?"))

print(gpt4.invoke("what is quantum physics?"))