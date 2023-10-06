from langchain.document_loaders import TextLoader

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(r"C:\Users\ejose\Documents\langchain\intro-to-vector-db-langchain\mediumblogs\mediumblog1.txt", encoding='utf-8')
    document = loader.load()
    print(document)