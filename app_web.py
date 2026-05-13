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
    # 1. Carrega o banco que o seu indexador.py criou
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    banco_vetorial = Chroma(persist_directory="banco_dados", embedding_function=embeddings)
    buscador = banco_vetorial.as_retriever(search_kwargs={"k": 3})

    # 2. Usa o Groq para velocidade extrema (Llama 3.1 70B ou 8B)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    template_prompt = """Você é um atendente de SAC educado e direto. 
    Responda à pergunta do usuário priorizando o contexto. 
    Se a resposta não estiver no contexto, seja prestativo, mas se não for possivel diga "Desculpe, não tenho essa informação".

    Contexto da Empresa:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": itemgetter("question") | buscador | (lambda docs: "\n\n".join(d.page_content for d in docs)),
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