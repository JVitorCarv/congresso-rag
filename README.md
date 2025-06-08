This repository contains a Jupyter Notebook that uses LangChain and OpenAI APIs to build a semantic search engine around journals from the Brazilian National Congress, applying RAG to uncover and keep track of political events. A specific journal from May 1, 2025 is used.

## Installation

To install the required Python packages, use [uv](https://github.com/astral-sh/uv):

1. Initialize the project (if not already done):

   ```
   uv init
   ```

2. Install dependencies from `pyproject.toml`:

   ```
   uv sync
   ```

## Usage

For this to work, you need to load an environment variable containing your OpenAI API key. If you are using this Jupyter Notebook in Visual Studio Code using the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), you can just create a `.env` file and place your API key there:

```
OPENAI_API_KEY=YOUR_API_KEY_HERE
```

The variable should be automatically loaded by the extension.

To run the Streamlit app, use the following command:

```
streamlit run main.py
```

## More Information

For more details, here's the related Medium post I wrote: [Keeping Up With Congress With Retrieval-Augmented Generation (RAG)](https://medium.com/@mateusriff/keeping-up-with-congress-with-retrieval-augmented-generation-rag-513387fa45b9)