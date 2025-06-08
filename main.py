import streamlit as st
from util.rag import RAG
from util.text_preprocessor import TextPreprocessor
import os


# This caching layer will avoid the document being re-processed
# after each query.
@st.cache_resource
def get_text_preprocessor(uploaded_file):
    return TextPreprocessor(uploaded_file)


@st.cache_data
def load_pdf_file(_preprocessor):
    return _preprocessor.load_file()


@st.cache_resource
def get_vector_store(_preprocessor):
    _preprocessor.split_text()
    return _preprocessor.load_embeddings()


def main():
    st.title("Congresso RAG")

    uploaded_file = st.file_uploader("Faça upload do PDF:", type="pdf")
    if not uploaded_file:
        return

    preprocessor = get_text_preprocessor(uploaded_file)

    with st.spinner("Carregando documento..."):
        docs = load_pdf_file(preprocessor)
        st.success(f"{len(docs)} páginas do PDF foram carregadas.")

    with st.spinner("Carregando embeddings..."):
        vector_store = get_vector_store(preprocessor)
    st.success("Documento indexado.")

    query = st.text_input("Digite sua pergunta:")

    if not query:
        return

    with st.spinner("Buscando resposta..."):
        instructions_path = os.path.join(os.getcwd(), "instructions.txt")
        rag = RAG(vector_store, instructions_path)

        try:
            response = rag.get_response(query)
            st.subheader("Resposta:")
            st.write(response.output_text)
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")


if __name__ == "__main__":
    main()
