import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configurações
PASTA_CONHECIMENTO = "base_conhecimento"
PASTA_DB = "banco_dados"

def criar_indice():
    print("🚀 Iniciando indexação de documentos...")
    
    documentos = []
    for arquivo in os.listdir(PASTA_CONHECIMENTO):
        caminho = os.path.join(PASTA_CONHECIMENTO, arquivo)
        try:
            if arquivo.endswith(".txt"):
                documentos.extend(TextLoader(caminho, encoding="utf-8").load())
            elif arquivo.endswith(".pdf"):
                documentos.extend(PyPDFLoader(caminho).load())
            elif arquivo.endswith(".docx"):
                documentos.extend(Docx2txtLoader(caminho).load())
            print(f"✅ Lido: {arquivo}")
        except Exception as e:
            print(f"❌ Erro em {arquivo}: {e}")

    if not documentos:
        print("⚠️ Nenhum documento encontrado!")
        return

    # Quebra em pedaços
    separador = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], # Força a quebrar primeiro por parágrafos
        chunk_size=1200,                    # Dobramos o tamanho do contexto
        chunk_overlap=200                   # Aumentamos a sobreposição para não cortar ideias no meio
    )
    pedacos = separador.split_documents(documentos)

    # Embeddings e persistência no disco
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"🧠 Gerando embeddings para {len(pedacos)} pedaços... Aguarde.")
    Chroma.from_documents(
        documents=pedacos, 
        embedding=embeddings, 
        persist_directory=PASTA_DB
    )
    print(f"✨ Sucesso! Banco de dados salvo em '{PASTA_DB}'")

if __name__ == "__main__":
    criar_indice()