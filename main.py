from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="6c3fead1-e388-4081-859f-dd8400d940b5", environment="gcp-starter")
import os
if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("/Users/ejose/Documents/langchain/intro-to-vector-db-langchain/mediumblogs/mediumblog1.txt",   encoding="utf-8",)
    document = loader.load()

    print(f"Type of 'document': {type(document)}")
    print(f"Content of 'document': {document}")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")
    # docsearch = Pinecone.from_documents(texts, embeddings, index_name=)
