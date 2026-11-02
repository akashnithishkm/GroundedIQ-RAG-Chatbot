# PDF RAG Chatbot

A Streamlit PDF question-answering app using Gemini embeddings, ChromaDB,
LangGraph, and selectable Gemini, OpenRouter, or Ollama generation.

## Requirements

- Python 3.10 or later
- A Google API key for PDF embeddings and retrieval
- Ollama installed separately only if you want to use local Ollama models
- An OpenRouter API key if you use OpenRouter (free models still require one)

## Install the Python dependencies

From this project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` contains Python packages only. Its `langchain-ollama`
package lets the app connect to Ollama; it cannot install the Ollama
application or download Ollama model files.

## Configure API keys

Run the app and enter keys in the **API keys** section in the sidebar. The
Google API key is required for PDF embeddings and retrieval. OpenRouter
always requires its own API key, including when using a free model, so enter
one if you select OpenRouter or enable automatic fallback to it. Ollama runs
locally and does not require an API key.

Keys are held in the active Streamlit session and are not written to project
files. Use the **Clear API keys** button to remove them from the session. Only
enter keys into an app instance you trust; a remotely hosted app's operator
can access keys submitted to it. Standalone `ingest.py` and `rag_graph.py`
commands prompt for the Google key without echoing it.

Never commit API keys. If a key has already been committed or shared, revoke
it with its provider and create a replacement.

## Install Ollama and its models (optional)

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download)
   and start the Ollama app. The local API should respond at
   `http://localhost:11434`.
2. In PowerShell, download each model you want to use:

   ```powershell
   ollama pull llama3.2:3b
   ollama pull phi3:mini
   ollama pull mistral:7b
   ```

   Each model is downloaded separately. You only need to pull the models you
   plan to use. Model files take additional disk space and are not included in
   this GitHub project or `requirements.txt`.
3. Confirm the installed models:

   ```powershell
   ollama list
   Invoke-RestMethod http://localhost:11434/api/tags
   ```

4. In the app sidebar, choose **Ollama** and select a model that appears in
   `ollama list`.

## Run the app

```powershell
streamlit run app.py
```

Upload PDFs in the app and process them, or place them in `documents/` and
use **Process PDFs already in documents/**. Already indexed local PDFs are
shown with their chunk counts, so you do not need to upload them again.

The `documents/` folder and Chroma database are local data and are excluded
from Git. Keep private or copyrighted PDFs on your machine; a fresh GitHub
clone starts with an empty documents folder and must index PDFs again after
you provide a Google API key.
