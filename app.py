from flask import Flask, request, jsonify
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# 1. Carrega o modelo de IA assim que o servidor inicia
print("Carregando modelo de IA...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modelo carregado com sucesso!")

URL_PRODUTOS_FIREBASE = "https://tsukadaestoqeuapp-f0cqd9fpfefwhage.brazilsouth-01.azurewebsites.net/api/produtos/"

# Variáveis globais para guardar os produtos na memória do servidor
DF_PRODUTOS = None
EMBEDDINGS_PRODUTOS = None

def inicializar_banco_de_dados():
    """Busca os produtos da API e gera os embeddings apenas UMA vez no início"""
    global DF_PRODUTOS, EMBEDDINGS_PRODUTOS
    try:
        print("Buscando produtos no Firebase e gerando mapa semântico...")
        response = requests.get(URL_PRODUTOS_FIREBASE, timeout=15)
        if response.status_code != 200:
            print("Erro ao acessar a API do Firebase.")
            return False
        
        dados = response.json()
        df = pd.DataFrame(dados)
        df = df.fillna("")
        
        df['metadados'] = (
            df['nome'].astype(str) + " " + 
            df['categoria'].astype(str) + " " + 
            df['descricao'].astype(str) + " " + 
            df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        )
        
        DF_PRODUTOS = df
        EMBEDDINGS_PRODUTOS = model.encode(df['metadados'].tolist(), convert_to_tensor=True)
        print("Banco de dados da IA inicializado com sucesso!")
        return True
    except Exception as e:
        print(f"Falha na inicialização: {e}")
        return False

@app.route('/recomendar', methods=['POST'])
def recomendar():
    global DF_PRODUTOS, EMBEDDINGS_PRODUTOS
    
    data = request.get_json()
    if not data or 'mensagem' not in data:
        return jsonify({"erro": "O campo 'mensagem' é obrigatório"}), 400
    
    input_usuario = data['mensagem']
    
    # Se por algum motivo falhou no início, tenta carregar agora
    if DF_PRODUTOS is None or EMBEDDINGS_PRODUTOS is None:
        if not inicializar_banco_de_dados():
            return jsonify({"erro": "O banco de dados da IA não está pronto"}), 500
    
    # Processa a busca semântica
    embedding_usuario = model.encode(input_usuario, convert_to_tensor=True)
    scores = util.cos_sim(embedding_usuario, EMBEDDINGS_PRODUTOS)[0]
    
    top_k = min(3, len(DF_PRODUTOS))
    indices_top = scores.argsort(descending=True)[:top_k]
    
    resultados = []
    for idx in indices_top:
        item_idx = idx.item()
        produto = DF_PRODUTOS.iloc[item_idx].to_dict()
        produto.pop('metadados', None)
        produto['ia_score'] = float(scores[idx].item()) 
        resultados.append(produto)
        
    return jsonify({"recomendacoes": resultados})

# Rota extra se mudar algo no Firebase e querer atualizar a IA sem reiniciar o servidor
@app.route('/atualizar_produtos', methods=['GET'])
def atualizar():
    if inicializar_banco_de_dados():
        return jsonify({"status": "IA atualizada com os novos produtos do Firebase!"}), 200
    return jsonify({"erro": "Não foi possível atualizar"}), 500

# Executa a carga inicial quando o Flask liga
inicializar_banco_de_dados()

if __name__ == '__main__':
    app.run(debug=True, port=5000)