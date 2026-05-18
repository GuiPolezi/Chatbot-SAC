# 🤖 Chatbot SAC — Atendimento Inteligente

> Chatbot de Serviço de Atendimento ao Cliente (SAC) com base de conhecimento personalizada, alimentado por IA via **Groq**, vetores armazenados no **Qdrant Cloud** e API hospedada no **Hugging Face Spaces**.

---

## 📋 Visão Geral

Este projeto implementa um chatbot de SAC que responde perguntas com base em uma **base de conhecimento própria**. Os documentos são indexados e armazenados como vetores no Qdrant Cloud. A cada consulta, o sistema recupera os trechos mais relevantes e os envia para a IA gerar uma resposta contextualizada.

### Stack utilizada

| Componente | Tecnologia |
|---|---|
| 🧠 Modelo de IA | [Groq API](https://groq.com/) |
| 🗄️ Banco Vetorial | [Qdrant Cloud](https://qdrant.tech/) |
| 🌐 API do sistema | [Hugging Face Spaces](https://huggingface.co/spaces) |
| 🐍 Runtime | Python + venv |

---

## 📁 Estrutura do Projeto

```
📦 chatbot-sac/
├── 📂 banco_conhecimento/       # ⚠️ Pasta local — NÃO está no GitHub
│   └── seus_arquivos.pdf/.txt/... 
├── indexador.py                 # Script para indexar a base de conhecimento
├── app.py                       # Aplicação principal / API
├── requirements.txt
└── README.md
```

> **⚠️ Atenção:** A pasta `banco_conhecimento/` **não está versionada** no repositório. Ela deve ser criada manualmente na máquina local antes de rodar o sistema.

---

## ⚙️ Configuração Inicial

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie e ative o ambiente virtual

```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes chaves:

```env
GROQ_API_KEY=sua_chave_groq
QDRANT_URL=sua_url_qdrant_cloud
QDRANT_API_KEY=sua_chave_qdrant
HUGGINGFACE_API_URL=url_do_seu_space
```

### 5. Crie a pasta da base de conhecimento

```bash
mkdir banco_conhecimento
```

Adicione seus arquivos de conhecimento dentro desta pasta (`.txt`).

---

## 🔄 Atualizando a Base de Conhecimento

Sempre que quiser adicionar ou atualizar informações que o bot utiliza para responder:

### Passo a passo

```
1. Ative o ambiente virtual
   └── venv\Scripts\activate

2. Adicione novos arquivos em:
   └── banco_conhecimento/

3. Rode o indexador:
   └── python indexador.py
```

O indexador irá processar os arquivos, gerar os embeddings e enviar os vetores atualizados para o **Qdrant Cloud**.

---

## 🚀 Executando o Projeto

```bash
# Com o venv ativado:
Local
- Em um terminal ative a API: uvicorn api:app --reload
- Em outro terminal rode a aplicação - Deve estar conectada com a api.
```
```
Remoto
Garanta que ja esteja configurado o Qdrant Cloud junto com a API da IA
- Suba a API do sistema para o Hugging Face Spaces
- Suba seu front-end para a web (Ele deve estar conectado com a api do sistema)
- Aproveite!
```

---

## 🏗️ Arquitetura

```
Usuário
   │
   ▼
┌──────────────┐      ┌──────────────────┐
│  Hugging Face │ ───► │  Busca vetorial  │
│  Spaces (API) │      │  (Qdrant Cloud)  │
└──────────────┘      └────────┬─────────┘
                                │
                       Contexto relevante
                                │
                                ▼
                       ┌────────────────┐
                       │   Groq API     │
                       │  (LLM / Chat)  │
                       └────────┬───────┘
                                │
                                ▼
                          Resposta ao usuário
```

---

## 📌 Observações Importantes

- A pasta `banco_conhecimento/` está listada no `.gitignore` — **nunca a versione** se contiver dados sensíveis.
- Sempre **reindexe após adicionar novos documentos** (`python indexador.py`).
- O Qdrant Cloud mantém os vetores persistidos em nuvem — não é necessário reindexar a cada inicialização, apenas quando a base de conhecimento mudar.

---

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie sua branch: `git checkout -b feature/minha-feature`
3. Commit suas mudanças: `git commit -m 'feat: minha feature'`
4. Push para a branch: `git push origin feature/minha-feature`
5. Abra um Pull Request

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.