import streamlit as st
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error" # Para ocultar logs do streamlit no console 
from dotenv import load_dotenv
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Carregando .env
load_dotenv()

# Configurações de Interface
st.set_page_config(page_title="SAC Inteligente - Alpha", page_icon="🤖")
st.title("🤖 Atendente Virtual (Alpha)")

# Carregar chave do Groq (pode colocar no .env ou segredos do Streamlit)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def carregar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    banco_vetorial = Chroma(persist_directory="banco_dados", embedding_function=embeddings)
    
    # 1. AUMENTAMOS O K PARA 6 (O bot terá o dobro de contexto para ler)
    buscador = banco_vetorial.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    # 2. PROMPT BLINDADO CONTRA CONFUSÕES
    template = """Você é um atendente de SAC. 
    Responda com base estritamente no contexto fornecido. 
    
    ATENÇÃO: Cada trecho de contexto possui o nome do [Arquivo de Origem]. 
    Preste muita atenção ao nome do arquivo para NÃO misturar regras, manuais ou informações de sistemas diferentes. 
    Se o cliente mencionar um sistema específico, priorize os trechos que vêm do arquivo desse sistema.

    Contexto:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # 3. FUNÇÃO QUE INJETA O NOME DO ARQUIVO ANTES DO TEXTO
    def juntar_textos_com_fonte(docs):
        textos_formatados = []
        for d in docs:
            # Pega o caminho do arquivo e extrai só o nome (ex: manual_sistema_A.pdf)
            caminho_completo = d.metadata.get('source', 'Desconhecido')
            nome_arquivo = os.path.basename(caminho_completo)
            
            # Cola o nome do arquivo junto com o texto para a IA ler
            textos_formatados.append(f"[Arquivo de Origem: {nome_arquivo}]\n{d.page_content}")
            
        return "\n\n---\n\n".join(textos_formatados)

    # 4. ATUALIZAMOS A CHAIN PARA USAR A NOVA FUNÇÃO
    chain = (
        {
            "context": itemgetter("question") | buscador | juntar_textos_com_fonte,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt | llm | StrOutputParser()
    )
    return chain

fluxo_rag = carregar_sistema()

# Memória da sessão (Histórico)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Exibir mensagens anteriores
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# Entrada do usuário
if prompt_user := st.chat_input("Como posso ajudar?"):
    st.chat_message("user").write(prompt_user)
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de dados..."):
            resposta = fluxo_rag.invoke({
                "question": prompt_user,
                "chat_history": st.session_state.chat_history
            })
            st.write(resposta)

    st.session_state.chat_history.append(HumanMessage(content=prompt_user))
    st.session_state.chat_history.append(AIMessage(content=resposta))