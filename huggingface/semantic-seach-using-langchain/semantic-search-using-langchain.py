######Loading Documents######
from langchain_community.document_loaders import PyPDFLoader

file_path = "data/attention-is-all-you-need.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
#print(f"{docs[0].page_content[:200]}\n")
#print(docs[0].metadata)


#####Splitting########
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))


#####Embeddings########
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
#doc_result = embeddings.embed_documents([all_splits]) # for a list of documents
#print(doc_result[:5])


####Vector Store#####
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search_with_score(
    "The encoder is composed of a stack of how many identical layers?"
)

print(results[0][0].page_content)   # direct index access for tupple unpacking
print(results[0][1])