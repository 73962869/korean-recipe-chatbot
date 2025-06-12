import streamlit as st
from recipe_llm import stream_ai_message
import uuid

st.set_page_config(page_title='🍲 한식 요리 추천 챗봇', page_icon='🍲')
st.title('🍲 냉장고를 부탁해 봇(한식)')

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

placeholder = '냉장고 속 재료를 입력하세요 (예: 김치, 두부, 계란)'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner('요리 추천 중...'):
        session_id = st.session_state.session_id
        ai_message = stream_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_message)
        st.session_state.message_list.append({'role': 'ai', 'content': ai_message})
