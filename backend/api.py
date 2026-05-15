import os
import re
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Imports leves no topo
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
load_dotenv()

# ==========================================
# MODELOS DE DADOS (PYDANTIC)
# ==========================================
class Mensagem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    pergunta: str
    historico: List[Mensagem] = []

class ChatResponse(BaseModel):
    resposta: str
    fontes: List[str]

# ==========================================
# INICIALIZAÇÃO DO APP
# ==========================================
app = FastAPI(title="API SAC Inteligente (Alpha)", description="Cérebro do chatbot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

buscador = None
fluxo_rag = None

def carregar_sistema():
    global buscador, fluxo_rag
    
    print("Iniciando importações pesadas e conexão com o banco...")
    
    # LAZY IMPORTS: Tira o peso da inicialização do servidor
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_groq import ChatGroq

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    cliente_qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    banco_vetorial = QdrantVectorStore(
        client=cliente_qdrant, 
        collection_name="Chatbot", 
        embedding=embeddings
    )
    
    buscador = banco_vetorial.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 40})
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    template = """Você é um atendente de SAC focado em ajudar o cliente com precisão. 

    REGRAS:
    1. Responda baseando-se ESTRITAMENTE no contexto fornecido abaixo.
    2. Cada trecho de informação possui o nome do [Arquivo de Origem]. Caso haja informações conflitantes, priorize responder o que faz mais sentido para a pergunta.
    3. Se a resposta para a pergunta não estiver no contexto, NÃO INVENTE. Diga: "Desculpe, não tenho essa informação exata na minha base de conhecimento no momento."
    4. EXCEÇÃO DE SAUDAÇÃO: Se a mensagem do usuário for apenas uma saudação (oi, olá, bom dia, boa tarde) ou um agradecimento, ignore o contexto e apenas responda de forma natural, educada e prestativa.

    Contexto:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    fluxo_rag = prompt | llm | StrOutputParser()
    print("Sistema carregado com sucesso!")

def juntar_textos_com_fonte(docs):
    textos_formatados = []
    for d in docs:
        nome_arquivo = os.path.basename(d.metadata.get('source', 'Desconhecido'))
        textos_formatados.append(f"[Arquivo de Origem: {nome_arquivo}]\n{d.page_content}")
    return "\n\n---\n\n".join(textos_formatados)

# ==========================================
# ROTA PRINCIPAL DA API
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global buscador, fluxo_rag
    
    if buscador is None or fluxo_rag is None:
        try:
            print("Carregando o sistema pela primeira vez...")
            carregar_sistema()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao inicializar IA: {str(e)}")
            
    try:
        chat_history_langchain = []
        for msg in request.historico:
            if msg.role == "user":
                chat_history_langchain.append(HumanMessage(content=msg.content))
            else:
                chat_history_langchain.append(AIMessage(content=msg.content))

        documentos_encontrados = buscador.invoke(request.pergunta)
        contexto_formatado = juntar_textos_com_fonte(documentos_encontrados)
        
        resposta = fluxo_rag.invoke({
            "context": contexto_formatado,
            "question": request.pergunta,
            "chat_history": chat_history_langchain
        })
        
        nomes_arquivos = list(set([os.path.basename(doc.metadata.get('source', 'Desconhecido')) for doc in documentos_encontrados]))

        if "Já registrei a instabilidade" in resposta:
            padrao_link = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            links_encontrados = re.findall(padrao_link, resposta)
            
            if links_encontrados:
                site_com_erro = links_encontrados[0].rstrip(".,!?")
                webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
                
                if webhook_url:
                    mensagem_discord = {
                        "content": f"🚨 **ALERTA DE INSTABILIDADE** 🚨\nO bot registrou erro no site: **{site_com_erro}**\nEquipe técnica, favor verificar!"
                    }
                    requests.post(webhook_url, json=mensagem_discord)
        
        return ChatResponse(resposta=resposta, fontes=nomes_arquivos)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))