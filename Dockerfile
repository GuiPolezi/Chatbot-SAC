# Usa uma imagem do Python leve e otimizada
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala as dependências do sistema necessárias
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copia e instala as bibliotecas do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o seu código para o servidor
COPY . .

# CRÍTICO PARA O HUGGING FACE: Cria pasta de cache com permissão total 
# para o modelo ser baixado sem dar erro de "Permission Denied"
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# O Hugging Face Spaces EXIGE que a aplicação rode na porta 7860
EXPOSE 7860

# Comando que liga o Uvicorn na porta correta
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]