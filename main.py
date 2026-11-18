
import os
import uuid
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-6-luna")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please configure your .env file.")
    st.stop()


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DwD Enterprise Chatbot",
    page_icon="💬",
    layout="centered"
)


# ==========================================
# 3. INITIALIZE LANGUAGE MODEL
# ==========================================

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY
)



# ==========================================
# 4. CREATE CHAT PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional customer assistance chatbot
        for DwD Enterprise.

        Your duties and responsibilities include:

        - Be polite and professional.
        - Give simple, clear and helpful answers.
        - Help customers with their questions.
        - Use previous conversation history when relevant.
        - Do not invent customer or order information.
        - Never provide unverified order details.
        - If you do not know something, clearly say so.
        - Keep responses relevant to DwD Enterprise.
        - Ask clarifying questions when necessary.
        - Thank customers for contacting DwD Enterprise.
        """
    ),

    # Include previous conversation history
    MessagesPlaceholder(variable_name="history"),

    # Current customer question
    ("human", "{question}")
])


# ==========================================
# 5. CREATE LANGCHAIN CHAIN
# ==========================================

parser = StrOutputParser()

chain = prompt | llm | parser


# ==========================================
# 6. CHAT HISTORY MANAGEMENT
# ==========================================

store = {}


def get_session_history(session_id: str):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# ==========================================
# 7. CREATE CHATBOT WITH MESSAGE HISTORY
# ==========================================

chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)


# ==========================================
# 8. INITIALIZE STREAMLIT SESSION STATE
# ==========================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


if "messages" not in st.session_state:

    st.session_state.messages = []


if "chat_ended" not in st.session_state:

    st.session_state.chat_ended = False


# ==========================================
# 9. CHATBOT HEADER
# ==========================================

st.title("💬 DwD Enterprise")

st.subheader("Customer Support Assistant")

st.write(
    "Welcome to DwD Enterprise! "
    "How can I assist you today?"
)

st.divider()


# ==========================================
# 10. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("⚙️ Chat Settings")

    st.write("DwD Enterprise Customer Support")

    st.caption(f"AI Model: {MODEL_NAME}")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        # Start a fresh conversation
        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.session_state.chat_ended = False

        st.rerun()

    if st.button("🚪 End Conversation", use_container_width=True):

        st.session_state.chat_ended = True

        st.rerun()

    st.divider()

    st.caption("Powered by Streamlit and LangChain")


# ==========================================
# 11. DISPLAY PREVIOUS CHAT MESSAGES
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 12. CHATBOT RESPONSE FUNCTION
# ==========================================

def get_chatbot_response(question):

    response = chat.invoke(
        {"question": question},

        config={
            "configurable": {
                "session_id": st.session_state.session_id
            }
        }
    )

    return response


# ==========================================
# 13. CHAT INPUT AND RESPONSE
# ==========================================

if st.session_state.chat_ended:

    st.success(
        "Thank you for contacting DwD Enterprise. "
        "Your conversation has ended. "
        "Please clear the conversation to start again."
    )

else:

    user_question = st.chat_input(
        "Type your message here..."
    )

    if user_question:

        # Display customer message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):

            st.markdown(user_question)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("DwD Assistant is typing..."):

                try:

                    response = get_chatbot_response(
                        user_question
                    )

                    st.markdown(response)

                    # Save AI response for display
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:

                    error_message = (
                        "Sorry, I encountered an error "
                        "while processing your request. "
                        "Please try again."
                    )

                    st.error(error_message)

                    st.session_state.messages.pop()

                    st.error(f"Technical details: {e}")
