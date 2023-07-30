import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')


from htmlTemplates import css, bot_template, user_template

def get_pdf_text(user_pdf_docs):
    text = ""
    for pdf in user_pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=PINECONE_INDEX)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    st.set_page_config(page_title="MSME SAHAI",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(":books: MSME SAHAI")
    user_question = st.text_input("hey, there. I am your MSME SAHAI:", placeholder="😄 Ask me anything about MSMEs")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("MSME Legal Compliance Database:")
        st.write("This is a database of legal compliance documents for MSMEs in India. The documents are from verified sources and are updated regularly.")

        st.subheader("Add more legal compliance documents:")
        user_pdf_docs = st.file_uploader(
            "Upload your PDFs from verified sources and click on 'Process'", accept_multiple_files=True, type="pdf")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(user_pdf_docs)
                print(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print(text_chunks)
                

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                print(vectorstore)
                print(type(vectorstore))

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        else:
            raw_text = []
            vectorstore = get_vectorstore(raw_text)
            st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
