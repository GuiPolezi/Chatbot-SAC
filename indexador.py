import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Configurações
PASTA_CONHECIMENTO = "base_conhecimento"
# PASTA_DB = "banco_dados"
NOME_COLECAO = "Chatbot"


def criar_indice():
    print("🚀 Iniciando indexação de documentos para a nuvem...")
    
    documentos = []
    for arquivo in os.listdir(PASTA_CONHECIMENTO):
        caminho = os.path.join(PASTA_CONHECIMENTO, arquivo)
        try:
            if arquivo.endswith(".txt"):
                documentos.extend(TextLoader(caminho, encoding="utf-8").load())
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
    
    # Conecta no Qdrant Cloud
    url_qdrant = os.getenv("QDRANT_URL")
    api_key_qdrant = os.getenv("QDRANT_API_KEY")

    # ADICIONE ESTA LINHA PARA TESTAR:
    print(f"URL Lida do .env: {url_qdrant}")

    print("☁️ Conectando ao Qdrant Cloud...")
    cliente_qdrant = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)

    # Verifica se a coleção (tabela) já existe. Se existir, deleta para atualizar com os novos dados.
    if cliente_qdrant.collection_exists(collection_name=NOME_COLECAO):
        print(f"🗑️ Deletando base antiga '{NOME_COLECAO}' para recriar...")
        cliente_qdrant.delete_collection(collection_name=NOME_COLECAO)


    # Cria a coleção nova, avisando o tamanho dos vetores do nosso modelo
    print(f"🏗️ Criando coleção '{NOME_COLECAO}'...")
    cliente_qdrant.create_collection(
        collection_name=NOME_COLECAO,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print(f"🧠 Enviando {len(pedacos)} vetores para a nuvem... Aguarde.")
    QdrantVectorStore.from_documents(
        pedacos,
        embeddings,
        url=url_qdrant,
        api_key=api_key_qdrant,
        collection_name=NOME_COLECAO,
    )
    print("✨ Sucesso! Banco de dados enviado e atualizado na nuvem!")

'''
    print(f"🧠 Gerando embeddings para {len(pedacos)} pedaços... Aguarde.")
    Chroma.from_documents(
        documents=pedacos, 
        embedding=embeddings, 
        persist_directory=PASTA_DB
    )
    print(f"✨ Sucesso! Banco de dados salvo em '{PASTA_DB}'")
'''

if __name__ == "__main__":
    criar_indice()