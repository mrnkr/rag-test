import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool # pyright: ignore[reportUnknownVariableType]
from langchain.agents import create_agent # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content")) # pyright: ignore[reportPrivateImportUsage]
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = InMemoryVectorStore(embeddings)
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") # pyright: ignore[reportUnknownMemberType]
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

model = init_chat_model(model="llama3.1", model_provider="ollama")
tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use only that tool to help answer user queries."
    "Your answers should be brief and to the point, the bare minimum to answer the questions posed."
)

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text # pyright: ignore[reportUnknownMemberType]
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context]) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

# agent = create_agent(model, tools, system_prompt=prompt) # pyright: ignore[reportUnknownVariableType]

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream( # pyright: ignore[reportUnknownMemberType]
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
