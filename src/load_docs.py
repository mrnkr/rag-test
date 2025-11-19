import sys
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content")) # pyright: ignore[reportPrivateImportUsage]
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    requests_kwargs={"timeout": 10}
)
try:
    docs = loader.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    sys.exit(1)

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
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
document_ids = vector_store.add_documents(documents=all_splits)
