from flask import Flask, render_template, request
import pandas as pd
import joblib
from io import BytesIO
import requests

app = Flask(__name__)

# URL do arquivo raw no GitHub
url_modelo = 'https://github.com/njocimar28/Perfil-Investidor/raw/3b455fcbcb185167780d9daafdb78ec6b70879ce/modelo_random_forest_ivestimento.pkl'

# Baixar o conteúdo do arquivo do modelo
response_modelo = requests.get(url_modelo)
if response_modelo.status_code == 200:
    # Carregar o modelo a partir do conteúdo baixado
    modelo_carregado = joblib.load(BytesIO(response_modelo.content))
else:
    raise Exception("Falha ao baixar o modelo. Verifique a URL.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        genero = int(request.form['genero'])
        estado_civil = int(request.form['estado_civil'])
        escolaridade = int(request.form['escolaridade'])
        regiao = int(request.form['regiao'])
        idade = float(request.form['idade'])
        valor_investido_acoes = float(request.form['valor_investido_acoes'])
        valor_investido_fii = float(request.form['valor_investido_fii'])
        valor_investido_dolar = float(request.form['valor_investido_dolar'])
        valor_investido_renda_fixa = float(request.form['valor_investido_renda_fixa'])
        
        valor_investido = (
            valor_investido_acoes +
            valor_investido_fii +
            valor_investido_dolar +
            valor_investido_renda_fixa
        )

        dados = {
            'genero': [genero],
            'escolaridade': [escolaridade],
            'idade' : [idade],
            'estado_civil': [estado_civil],
            'regiao': [regiao],
            'valor_investido': [valor_investido],
            'valor_investido_acoes': [valor_investido_acoes],
            'valor_investido_fii': [valor_investido_fii],
            'valor_investido_dolar': [valor_investido_dolar],
            'valor_investido_renda_fixa': [valor_investido_renda_fixa]
        }

        # Crie um DataFrame
        df = pd.DataFrame(dados)

        # Faça as previsões
        novas_previsoes = modelo_carregado.predict(df)

        return render_template('result.html', previsoes=novas_previsoes)

if __name__ == '__main__':
    app.run(debug=True)