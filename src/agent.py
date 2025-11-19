from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool # pyright: ignore[reportUnknownVariableType]
from langchain.agents import create_agent # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_chroma import Chroma
from langchain.agents.structured_output import ToolStrategy

from response_dto import ResponseDto

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

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


ui_agent = create_agent(model, tools=[], middleware=[prompt_with_context]) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
agent = create_agent(model, tools=[], middleware=[prompt_with_context], response_format=ToolStrategy(ResponseDto)) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

# agent = create_agent(model, tools, system_prompt=prompt) # pyright: ignore[reportUnknownVariableType]
