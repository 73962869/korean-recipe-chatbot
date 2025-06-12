import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# CSV 파일 불러오기
df = pd.read_csv("예제_한식레시피.csv")

# LangChain용 Document 리스트로 변환
docs = [
    Document(page_content=f"요리명: {row['요리명']}\n재료: {row['재료']}\n조리방법: {row['조리방법']}")
    for _, row in df.iterrows()
]

# 분할 + 임베딩 + 인덱스 저장
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embedding)
db.save_local("recipe_index")
print("✅ 인덱스 저장 완료!")
