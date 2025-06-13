import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS


load_dotenv()


answer_examples = [
    {
        "input": "계란과 김치로 만들 수 있는 요리는?",
        "answer": "김치볶음밥을 추천합니다.\n- 요리명: 김치볶음밥\n- 재료: 김치, 밥, 계란, 참기름\n- 조리방법: 김치를 볶고 밥과 계란을 함께 볶아 완성합니다."
    },
    {
        "input": "두부랑 고추장 있어. 뭐 만들 수 있어?",
        "answer": "두부조림이 좋아요.\n- 요리명: 두부조림\n- 재료: 두부, 고추장, 마늘, 간장\n- 조리방법: 두부를 노릇하게 굽고 양념에 졸여줍니다."
    }
]

# 세션 히스토리 저장소
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# LLM 불러오기
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

# FAISS 벡터스토어 로딩
def load_vectorstore():
    embedding = OpenAIEmbeddings()
    db = FAISS.load_local("recipe_index", embedding, allow_dangerous_deserialization=True)
    return db

# history aware retriever
def build_history_aware_retriever(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 히스토리를 참고하여 독립된 질문으로 바꿔주세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, retriever, contextualize_prompt)

# few-shot 예시 prompt 생성
def bulid_few_shot_examples():
    example_prompt = PromptTemplate.from_template("질문: {input}\n\n답변: {answer}")
    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,
        example_prompt=example_prompt,
        prefix="다음은 요리 추천 예시입니다.",
        suffix="질문: {input}",
        input_variables=["input"]
    )
    return few_shot_prompt.format(input="{input}")

# QA Prompt 구성
def build_qa_prompt():
    system_prompt = (
        "[identity]\n"
        "- 당신은 한식 요리 추천 전문가이자 50년 한식요리장인입니다.\n"
        "- 사용자가 가진 재료에 맞춰 요리를 추천하세요.\n"
        "- 반드시 요리명, 재료, 조리방법 항목으로 나누어 답변하세요.\n"
        "- 요리 방법은 상세하고 자세히 설명해주세요.\n\n"
        "[context]\n{context}"
    )
    formatted_few_shot = bulid_few_shot_examples()
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("assistant", formatted_few_shot),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

# 전체 RAG 체인 구성
def build_conversational_chain():
    llm = load_llm()
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 2})
    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    qa_prompt = build_qa_prompt()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick("answer")


def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()
    return qa_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}}
    )
