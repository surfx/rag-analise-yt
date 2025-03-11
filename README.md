# Créditos

Todos os créditos à [
Tony Kipkemboi](https://www.youtube.com/@tonykipkemboi) e seu vídeo [How to chat with your PDFs using local Large Language Models [Ollama RAG]](https://www.youtube.com/watch?v=ztBJqzBU5kc&t=352s). Grato ao entusiasmo e partilhar seu conhecimento


# Python

```bash
curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

## C++

Instale o [vs_BuildTools.exe](https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/)

# UV

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.13.2
cd E:\programas\ia\virtual_environment
uv venv --python 3.13.2 my_env
my_env\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -U ipykernel tqdm numpy sympy chromadb protobuf==3.20.3 docling
uv pip install -U unstructured langchain langchain-community langchain_ollama "unstructured[all-docs]" ipywidgets


```

Para o notebook: `"E:\programas\ia\virtual_environment\my_env\Scripts\python.exe"`


## link simbolico (cmd)

```bash
MKLINK /D D:\meus_documentos\workspace\ia\rag\rag002\my_env E:\programas\ia\virtual_environment\my_env
```

## Chromadb

- [http://localhost:8000/](http://localhost:8000/)

```bash
chroma run --host localhost --port 8000 --path ./my_chroma_data
```

## Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull nomic-embed-text
ollama run llama3.2
ollama run deepseek-r1
```

# referencias

- [How to chat with your PDFs using local Large Language Models [Ollama RAG]](https://www.youtube.com/watch?v=ztBJqzBU5kc)
- [Chat with PDF locally using Ollama + LangChain](https://github.com/tonykipkemboi/ollama_pdf_rag/tree/main)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
- [llama3.2](https://ollama.com/library/llama3.2)
- [deepseek-r1](https://ollama.com/library/deepseek-r1)
- [pytorch](https://pytorch.org/get-started/locally/)
