from langchain.document_loaders import TextLoader
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()

pinecone.init(api_key="a7471a18-457f-4a3e-91f6-5795765ec681", environment="ej-gcp")
index = pinecone.Index("medium-blogs-embeddings-index")

pinecone.init(
    api_key="a7471a18-457f-4a3e-91f6-5795765ec681",
    environment="gcp-starter",
)

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(
        "/Users/ejose/Documents/langchain/intro-to-vector-db-langchain/mediumblogs/mediumblog1.txt",
        encoding="utf-8",
    )
    document = loader.load()

    print(f"Type of 'document': {type(document)}")
    print(f"Content of 'document': {document}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type_kwargs="stuff", retriever=docsearch.as_retriever(), return_source_documents=True
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner."
    result = qa({"query":query})
    print(result)
