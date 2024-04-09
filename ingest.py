from langchain_community.document_loaders.docusaurus import DocusaurusLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import settings


embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
vectorstore = Chroma(embedding_function=embeddings, persist_directory='./chroma_db')

# loader = DocusaurusLoader(
#     "https://python.langchain.com",
#     filter_urls=[
#         "https://python.langchain.com/docs/expression_language/"
#     ],
# )
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('../', glob="**/*.md", show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunked_docs = text_splitter.split_documents(documents)

vectorstore.add_documents(chunked_docs)
