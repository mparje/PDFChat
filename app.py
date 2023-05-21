from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path

import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
import glob

api_key = os.getenv('OPENAI_API_KEY')
qa_chain = None

async def main():

    async def storeDocEmbeds(file, filename):

        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

    async def getDocEmbeds(file, filename):

        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)

        with open(filename + ".pkl", "rb") as f:
            vectors = pickle.load(f)

        return vectors

    async def conversational_chat(query, chain):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Creando la interfaz del chatbot
    st.title("PDFChat")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    # Directorio que contiene los archivos PDF
    pdf_directory = "PDF"

    # Obtener la lista de archivos PDF en el directorio
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

    if pdf_files:
        st.write("Archivos PDF encontrados en la carpeta:")
        for pdf_file in pdf_files:
            st.write(pdf_file)

        selected_file = st.selectbox("Selecciona un archivo", pdf_files)

        if st.button("Procesar archivo"):
            with st.spinner("Procesando..."):
                file_name = Path(selected_file).stem
                vectors = await getDocEmbeds(selected_file, file_name)
                global qa_chain
                qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"),
                                                          retriever=vectors.as_retriever(),
                                                          return_source_documents=True)

            st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["¡Bienvenido! Ahora puedes hacer preguntas sobre el documento"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["¡Hola!"]

        # Contenedor para el historial del chat
        response_container = st.container()

        # Contenedor para el cuadro de texto
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Consulta:",
                                           placeholder="Por ejemplo: Resume el contenido del documento en unas pocas frases",
                                           key='input')
                submit_button = st.form_submit_button(label='Enviar')

            if submit_button and user_input:
                output = await conversational_chat(user_input, qa_chain)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    asyncio.run(main())
