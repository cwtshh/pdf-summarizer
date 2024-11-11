from langchain_ollama import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import EmbeddingsClusteringFilter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceBgeEmbeddings

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0
    )
    texts = text_splitter.split_documents(pages)
    return texts

def summarize_pdf(file, llm, embedding):
    filter = EmbeddingsClusteringFilter(embeddings=embedding, num_clusters=5)
    texts = extract_text_from_pdf(file)

    try:
        result = filter.transform_documents(documents=texts)
        summarize_chain = load_summarize_chain(llm, chain_type='stuff')
        summary = summarize_chain.run(result)
        return summary
    except Exception as e:
        return str(e)
    

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = { "device": "cuda" }
encode_kwargs = { "normalize_embeddings": True }

# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.5,
    
)

print(summarize_pdf("pdf-summarizer/dom-casmurro.pdf", llm, embeddings))

