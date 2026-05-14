import streamlit as st
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Carregando .env
load_dotenv()

st.set_page_config(page_title="SAC Inteligente - Alpha", page_icon="🤖")
st.title("🤖 Atendente Virtual (Alpha)")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def carregar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    banco_vetorial = Chroma(persist_directory="banco_dados", embedding_function=embeddings)
    
    # K=6 para dar mais visão à IA
    buscador = banco_vetorial.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    template = """Você é um atendente de SAC. Responda com base estritamente no contexto fornecido. 
    ATENÇÃO: Cada trecho possui o nome do [Arquivo de Origem]. 
    Preste atenção para não misturar informações de sistemas diferentes.

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

    # Salva na memória
    st.session_state.chat_history.append(HumanMessage(content=prompt_user))
    st.session_state.chat_history.append(AIMessage(content=resposta))