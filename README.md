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
uv venv --python 3.13.2 my_env_3129
my_env_3129\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -U ipykernel tqdm numpy sympy chromadb protobuf==3.20.3 docling
uv pip install -U unstructured langchain langchain-community langchain_ollama langchain_chroma "unstructured[all-docs]" ipywidgets
uv pip install -U pytesseract
```

Para o notebook: `"E:\programas\ia\virtual_environment\my_env_3129\Scripts\python.exe"`


## link simbolico (cmd)

```bash
MKLINK /D D:\meus_documentos\workspace\ia\rag\rag002\my_env_3129 E:\programas\ia\virtual_environment\my_env_3129
```

# tesserocr

- [tesserocr-windows_build](https://github.com/simonflueckiger/tesserocr-windows_build)
- [tesserocr-windows_build releases](https://github.com/simonflueckiger/tesserocr-windows_build/releases)

```bash
cd E:\programas\ia\virtual_environment
my_env_3129\Scripts\activate
uv pip install https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp312-cp312-win_amd64.whl
```

## tesseract releases

[tesseract releases](https://github.com/tesseract-ocr/tesseract/releases)

descompactar e add ao PATH do Windows: `E:\programas\ia\Tesseract-OCR\tessdata`

## [tessdata github](https://github.com/tesseract-ocr/tessdata)

Faça o download de [tessdata](https://github.com/tesseract-ocr/tessdata/archive/refs/heads/main.zip) e descompacte em: `E:\programas\ia\Tesseract-OCR\tessdata`

Windows PATH: 
- TESSDATA_PREFIX : E:\programas\ia\Tesseract-OCR\tessdata


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
- [Ollama Embeddings](https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/)
- [Chroma Persistent Client](https://docs.trychroma.com/docs/run-chroma/persistent-client)
- [tesseract releases](https://github.com/tesseract-ocr/tesseract/releases)
