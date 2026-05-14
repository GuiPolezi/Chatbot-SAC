import streamlit as st
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# imports para integração com discord - webhook
import re
import requests

# Carregando .env
load_dotenv()

st.set_page_config(page_title="SAC Inteligente - Alpha", page_icon="🤖")
st.title("🤖 Atendente Virtual (Alpha)")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def carregar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    '''
    # 1. Pegamos o caminho absoluto da pasta do seu projeto
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Juntamos com o nome da pasta do banco de dados
    caminho_absoluto_db = os.path.join(diretorio_atual, "banco_dados")
    
    # 3. Passamos esse caminho fixo para o Chroma
    banco_vetorial = Chroma(persist_directory=caminho_absoluto_db, embedding_function=embeddings)
    
    buscador = banco_vetorial.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40}
    )
    '''

    # Conecta no Qdrant usando as variáveis de ambiente
    url_qdrant = os.getenv("QDRANT_URL")
    api_key_qdrant = os.getenv("QDRANT_API_KEY")
    
    cliente_qdrant = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)
    
    # Busca a coleção que criamos no indexador
    banco_vetorial = QdrantVectorStore(
        client=cliente_qdrant, 
        collection_name="Chatbot", 
        embedding=embeddings
    )
    
    buscador = banco_vetorial.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40}
    )


    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    template = """Você é um atendente de SAC focado em ajudar o cliente com precisão. 

    REGRAS:
    1. Responda baseando-se ESTRITAMENTE no contexto fornecido abaixo.
    2. Cada trecho de informação possui o nome do [Arquivo de Origem]. Caso haja informações conflitantes, priorize responder o que faz mais sentido para a pergunta, ou explique que existem cenários diferentes dependendo do produto/sistema.
    3. Se a resposta para a pergunta não estiver no contexto, NÃO INVENTE. Diga: "Desculpe, não tenho essa informação exata na minha base de conhecimento no momento."

    Contexto:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # A corrente agora é mais simples: só liga o prompt ao cérebro
    chain = prompt | llm | StrOutputParser()
    
    return buscador, chain # Retornamos os dois separados!

# Inicializa o buscador e a IA
buscador, fluxo_rag = carregar_sistema()

# Função auxiliar para formatar os textos
def juntar_textos_com_fonte(docs):
    textos_formatados = []
    for d in docs:
        nome_arquivo = os.path.basename(d.metadata.get('source', 'Desconhecido'))
        textos_formatados.append(f"[Arquivo de Origem: {nome_arquivo}]\n{d.page_content}")
    return "\n\n---\n\n".join(textos_formatados)

# ==========================================
# INTERFACE DE CONVERSA E MEMÓRIA
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

if prompt_user := st.chat_input("Como posso ajudar?"):
    st.chat_message("user").write(prompt_user)
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de dados..."):
            
            # 1. BUSCA: Achamos os documentos ANTES de chamar a IA
            documentos_encontrados = buscador.invoke(prompt_user)
            
            # 2. FORMATAÇÃO: Colocamos os "crachás" nos textos
            contexto_formatado = juntar_textos_com_fonte(documentos_encontrados)
            
            # 3. IA: Enviamos tudo mastigado para o modelo processar
            resposta = fluxo_rag.invoke({
                "context": contexto_formatado,
                "question": prompt_user,
                "chat_history": st.session_state.chat_history
            })
            
            # Escreve a resposta
            st.write(resposta)
            
            # 4. FONTES: Cria uma lista com o nome dos arquivos (usamos 'set' para não repetir nomes)
            nomes_arquivos = set([os.path.basename(doc.metadata.get('source', 'Desconhecido')) for doc in documentos_encontrados])
            
            # Exibe as fontes em um menu expansível bonito
            if nomes_arquivos:
                with st.expander("📚 Fontes consultadas"):
                    for nome in nomes_arquivos:
                        st.markdown(f"- `{nome}`")


           # ==========================================
            # INTEGRAÇÃO COM O DISCORD
            # ==========================================
            # Se a IA validou o problema e enviou a mensagem do Passo 3:
            if "Já registrei a instabilidade" in resposta:
                # Procura o link do site NA RESPOSTA DA IA (ela tem a memória de tudo)
                padrao_link = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
                links_encontrados = re.findall(padrao_link, resposta)
                
                if links_encontrados:
                    # Limpa possíveis pontuações que grudam no link (como pontos finais ou vírgulas)
                    site_com_erro = links_encontrados[0].rstrip(".,!?")
                    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
                    
                    if webhook_url:
                        mensagem_discord = {
                            "content": f"🚨 **ALERTA DE INSTABILIDADE** 🚨\nO bot registrou erro no site: **{site_com_erro}**\nEquipe técnica, favor verificar!"
                        }
                        requests.post(webhook_url, json=mensagem_discord)
            # ==========================================

    # Salva na memória
    st.session_state.chat_history.append(HumanMessage(content=prompt_user))
    st.session_state.chat_history.append(AIMessage(content=resposta))