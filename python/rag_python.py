# cd E:\programas\ia\virtual_environment && my_env_3129\Scripts\activate
# uv run D:\meus_documentos\workspace\ia\rag\rag002\python\rag_python.py


PATH_ARQUIVOS = r"data"
LANG = "por" # por eng
# persist_directory = "chroma/chroma_db"  # Diretório onde o banco de dados será salvo
persist_directory = r"D:\meus_documentos\workspace\ia\rag\rag002\chroma\chroma_db"

# ------------------------------
# torch
# ------------------------------
import torch

def torch_init():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# torch_init()

# ------------------------------
# docling
# ------------------------------
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions, TesseractOcrOptions, OcrMacOptions
from docling.datamodel.settings import settings

IMAGE_RESOLUTION_SCALE = 2.0

# Define pipeline options for PDF processing
pipeline_options = PdfPipelineOptions(
    do_table_structure=True,  # Enable table structure detection
    do_ocr=True,  # Enable OCR
    # full page ocr and language selection
    #ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),  # Use EasyOCR for OCR
    ocr_options=TesseractOcrOptions(force_full_page_ocr=True, lang=[LANG]),  # Uncomment to use Tesseract for OCR
    #ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=['en-US']),
    table_structure_options=dict(
        do_cell_matching=False,  # Use text cells predicted from table structure model
        mode=TableFormerMode.ACCURATE  # Use more accurate TableFormer model
    ),
    generate_page_images=True,  # Enable page image generation
    generate_picture_images=True,  # Enable picture image generation
    images_scale=IMAGE_RESOLUTION_SCALE, # Set image resolution scale (scale=1 corresponds to a standard 72 DPI image)
)

# Initialize the DocumentConverter with the specified pipeline options
doc_converter_global = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# ------------------------------
# análise de documentos
# ------------------------------

import os

def separar_arquivos(diretorio):
    """
    Varre um diretório e suas subpastas, separando arquivos de imagem de outros tipos de arquivo.

    Args:
        diretorio (str): O caminho do diretório a ser varrido.

    Returns:
        tuple: Uma tupla contendo duas listas: imagens e documentos.
    """

    imagens = []
    documentos = []

    extensoes_imagens = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Adicione outras extensões se necessário

    for raiz, subpastas, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(raiz, arquivo)
            nome_arquivo, extensao = os.path.splitext(arquivo)
            extensao = extensao.lower()

            if extensao in extensoes_imagens:
                imagens.append(caminho_arquivo) #adiciona o caminho completo
            else:
                documentos.append(caminho_arquivo) #adiciona o caminho completo

    return imagens, documentos

#imagens, documentos = separar_arquivos(PATH_ARQUIVOS)

# print("Imagens:")
# for imagem in imagens: print(imagem)

# print("\nDocumentos:")
# for documento in documentos: print(documento)

# ------------------------------
# chromadb. obs: precisa do ollama executando
# ------------------------------
import hashlib
import os
import chromadb
from chromadb.config import Settings
from langchain.docstore.document import Document
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings 
from pathlib import Path

def generate_id_filename(filename, page_index):
    filename = os.path.basename(filename)
    base_id = hashlib.sha256(filename.encode()).hexdigest()
    return f"{base_id}_{page_index}"    

def generate_id(document, page_index):
    """Gera um ID único baseado no nome do arquivo e no índice da página."""
    source = document.metadata['source']
    return generate_id_filename(os.path.basename(source), page_index)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
chroma_client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
COLLECTION_NAME = "local-rag"

def chroma_indexing(path_arquivos=PATH_ARQUIVOS, collection_name=COLLECTION_NAME, embedding_model=embedding_model, chroma_client=chroma_client):
    """Indexa chunks em lote no ChromaDB."""

    imagens, documentos = separar_arquivos(path_arquivos)

    collection = chroma_client.get_or_create_collection(name=collection_name)

    for imagem in imagens:
        id_aux = generate_id_filename(imagem, 0)
        results = collection.get(ids=[id_aux])
        if results['ids'] and id_aux in results['ids']:
            print(f"Documento com ID {id_aux} | {os.path.basename(imagem)} já existe na coleção.")
            continue
        
        chroma_indexing_batch(get_chunks_image(imagem), collection, embedding_model)

    for documento in documentos:
        id_aux = generate_id_filename(documento, 0)
        results = collection.get(ids=[id_aux])
        if results['ids'] and id_aux in results['ids']: 
            print(f"Documento com ID {id_aux} | {os.path.basename(documento)} já existe na coleção.")
            continue

        chroma_indexing_batch(get_chunks_doc(documento), collection, embedding_model)

    return collection
    

def chroma_indexing_batch(chunks, collection=None, embedding_model=embedding_model):
    """Indexa chunks em lote no ChromaDB."""

    if not chunks or not collection:
        print('Sem chunks e/ou collection is null')
        return

    documents_to_add = []
    ids_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []

    for i, chunk in enumerate(chunks):
        document_id = generate_id(chunk, i)
        results = collection.get(ids=[document_id])

        if results['ids'] and document_id in results['ids']:
            print(f"Documento com ID {document_id} já existe na coleção.")
            continue

        embedding = embedding_model.embed_documents([chunk.page_content])[0]

        documents_to_add.append(chunk.page_content)
        ids_to_add.append(document_id)
        embeddings_to_add.append(embedding)
        metadatas_to_add.append(chunk.metadata)

    if documents_to_add:
        collection.add(documents=documents_to_add, ids=ids_to_add, embeddings=embeddings_to_add, metadatas=metadatas_to_add)
        print(f"Adicionados {len(documents_to_add)} documentos em lote.")

    if chunks:
        first_chunk_id = generate_id(chunks[0], 0)
        print(f"ID do primeiro chunk: {first_chunk_id}")

# TODO: verificar se está realmente funcionando
def delete_chroma_collection(collection_name=COLLECTION_NAME, chroma_client=None):
    if not collection_name or not chroma_client: return False
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' excluído com sucesso.")
        return True
    except Exception as e:
        print(f"Ocorreu um erro ao excluir o collection: {e}")
    return False

def reset_chroma(chroma_client=None):
    if not chroma_client: return False
    chroma_client.reset()
    return True

# ------------------------------
# document chunks
# ------------------------------

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# display(result.document.export_to_markdown())
from langchain_core.documents import Document

# img
import pytesseract
from PIL import Image
#pytesseract.pytesseract.tesseract_cmd = r"E:\programas\ia\Tesseract-OCR\tesseract.exe"
#end img

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

#local_path = r"data\pdfs\monopoly.pdf"
def get_chunks_doc(local_path):
    result = doc_converter_global.convert(Path(local_path))
    documento = Document(page_content=result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED), metadata={"source": local_path})
    chunks = text_splitter.split_documents([documento])
    return chunks


def get_chunks_image(local_path):
    image = Image.open(local_path)
    extracted_text = pytesseract.image_to_string(image, lang=LANG)

    documento = Document(page_content=extracted_text, metadata={"source": local_path})
    chunks = text_splitter.split_documents([documento])
    return chunks

# ------------------------------
# chroma indexing
# ------------------------------
# collection_name = "local-rag"

# for imagem in imagens:
#     chunks = get_chunks_image(imagem)
#     if (chunks is None or len(chunks) <= 0): continue
#     chroma_indexing(chunks, collection_name)

# for documento in documentos:
#     chunks = get_chunks_doc(documento)
#     if (chunks is None or len(chunks) <= 0): continue
#     chroma_indexing(chunks, collection_name)

# print("Documentos indexados")

# def indexar_arquivos_chromadb(path_arquivos=PATH_ARQUIVOS, collection_name=COLLECTION_NAME):
#     imagens, documentos = separar_arquivos(path_arquivos)
#     for imagem in imagens:
#         chunks = get_chunks_image(imagem)
#         if chunks:
#             chroma_indexing_batch(chunks, collection_name, embedding_model, chroma_client)

#     for documento in documentos:
#         chunks = get_chunks_doc(documento)
#         if chunks:
#             chroma_indexing_batch(chunks, collection_name, embedding_model, chroma_client)

#     print("Documentos indexados")

# chunks_list = []

# for imagem in imagens:
#     chunks = get_chunks_image(imagem)
#     if chunks: chunks_list.append(chunks)

# for documento in documentos:
#     chunks = get_chunks_doc(documento)
#     if chunks: chunks_list.append(chunks)

# chroma_indexing_batch(chunks_list, collection_name, embedding_model, chroma_client)

# print("Documentos indexados")

#indexar_arquivos_chromadb()

# ------------------------------
# consultas
# ------------------------------

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
local_model = "llama3.2"
local_model = "deepseek-r1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
    versões diferentes da pergunta do usuário fornecida para recuperar documentos relevantes de
    um banco de dados vetorial. Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu
    objetivo é ajudar o usuário a superar algumas das limitações da pesquisa de similaridade 
    baseada em distância. Forneça essas perguntas alternativas separadas por quebras de linha.
    Responda sempre no idioma português Brasil, salvo siglas, termos e frases específicos de 
    cada idioma
    Pergunta original: {question}""",
)

from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
vector_db = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Responda à pergunta com base SOMENTE no seguinte contexto:
{context}
Pergunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def do_question(prompt):
    return chain.invoke(prompt)

#print(do_question('o que é o WEF ?'))


# ------------------------------
# FLASK
# ------------------------------
import concurrent.futures
from flask import Flask, request, jsonify

app = Flask(__name__)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Ajuste max_workers conforme necessário

@app.route('/indexarChromaDB', methods=['POST'])
def indexar_chromadb():
    """
    Indexa os arquivos de uma pasta no chroma db
    """
    path_arquivos = request.data.decode('utf-8')  # Obtém o corpo da requisição como string

    collection_name = request.args.get('collection_name')
    collection_name = COLLECTION_NAME if not collection_name else collection_name

    if not path_arquivos: return "Nenhum parâmetro 'path_arquivos' fornecido."
    # chroma_indexing(path_arquivos, collection_name, embedding_model, chroma_client)
    executor.submit(chroma_indexing, path_arquivos, collection_name, embedding_model, chroma_client)
    return "Indexação iniciada em segundo plano."

@app.route('/doQuestion', methods=['GET'])
def do_question_llm():
    prompt = request.args.get('prompt')
    return "Nenhum parâmetro 'prompt' fornecido." if not prompt else do_question(prompt)

@app.route('/deleteCollection', methods=['DELETE'])
def delete_collection():
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({"success": False, "message": "Collection name not provided"})
    result = delete_chroma_collection(collection_name, chroma_client)
    return jsonify({"success": result, "message": "Collection deleted successfully" if result else "Collection deletion failed"})

@app.route('/resetChroma', methods=['GET'])
def reset_chromadb():
    result = reset_chroma(chroma_client)
    return jsonify({"success": result, "message": "Chroma resetado com sucesso" if result else "Erro ao resetar Chroma"})


if __name__ == '__main__':
    torch_init()
    app.run(debug=True)

# cd E:\programas\ia\virtual_environment && my_env_3129\Scripts\activate
# uv run D:\meus_documentos\workspace\ia\rag\rag002\python\rag_python.py

# curls
#  curl -X POST -H "Content-Type: text/plain" -d "D:\meus_documentos\workspace\ia\rag\rag002\data" http://127.0.0.1:5000/indexarChromaDB?collection_name=local-rag
#  curl "http://127.0.0.1:5000/doQuestion?prompt=Como+jogar+monopoly+%3F"
#  curl -X DELETE "http://127.0.0.1:5000/deleteCollection?collection_name=local-rag"
#  curl "http://127.0.0.1:5000/resetChroma"


