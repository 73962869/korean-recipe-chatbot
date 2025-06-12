import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

def get_database():
    embedding = OpenAIEmbeddings()
    database = FAISS.load_local(
        "recipe_index",
        embedding,
        allow_dangerous_deserialization=True  # 🔥 요거 추가
    )
    return database


def get_history_retriever(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and user query, create a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, contextualize_prompt)

def get_qa_prompt():
    system_prompt = (
        """
        [identity]
        - 당신은 한식 요리 추천 전문가입니다.
        - 사용자가 가진 재료에 맞춰 요리를 추천하세요.
        - 반드시 요리명, 재료, 조리방법 항목으로 나누어 답변하세요.
        - 요리 방법을 상세하고 자세히 설명해주세요.

        [context]
        {context}
        """
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def build_conversational_chain():
    llm = get_llm()
    db = get_database()
    retriever = db.as_retriever(search_kwargs={'k': 3})

    history_aware_retriever = get_history_retriever(llm, retriever)
    qa_prompt = get_qa_prompt()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()
    ai_message = qa_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    return ai_message