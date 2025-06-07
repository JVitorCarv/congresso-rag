import streamlit as st
from util.rag import RAG
from util.text_preprocessor import TextPreprocessor
import os


def main():
    st.title("Congresso RAG")

    uploaded_file = st.file_uploader("Faça upload do PDF:", type="pdf")
    if not uploaded_file:
        return

    query = st.text_input("Digite sua pergunta:")

    with st.spinner("Carregando documento..."):
        text_preprocessor = TextPreprocessor(uploaded_file)
        docs = text_preprocessor.load_file()
        st.success(f"{len(docs)} páginas do PDF foram carregadas.")

    with st.spinner("Carregando embeddings..."):
        vector_store = text_preprocessor.load_embeddings()
    st.success("Documento indexado.")

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
