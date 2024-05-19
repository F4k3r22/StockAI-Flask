# Derechos de autor (c) F4k3r22, Fredy Rivera. Todos los derechos reservados.
# Licencia y terminos: RiveraAICloseLicense(2006) para el Modelo de NLP StockAI

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request

app = Flask(__name__)

# Definir el modelo Transformer
class Transformer(nn.Module):
    def __init__(self, input_size, prediction_days, dim_feedforward):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward*2)
        self.fc3 = nn.Linear(dim_feedforward*2, dim_feedforward*4)
        self.fc4 = nn.Linear(dim_feedforward*4, dim_feedforward*8)
        self.fc5 = nn.Linear(dim_feedforward*8, dim_feedforward*16)
        self.fc6 = nn.Linear(dim_feedforward*16, dim_feedforward*32)
        self.fc7 = nn.Linear(dim_feedforward*32, dim_feedforward*64)
        self.fc8 = nn.Linear(dim_feedforward*64, dim_feedforward*128)
        self.fc9 = nn.Linear(dim_feedforward*128, dim_feedforward*256)
        self.fc10 = nn.Linear(dim_feedforward*256, prediction_days)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc9(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc10(x)
        return x

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html', predictions=None)

# Ruta para predecir
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    days = int(request.form['days'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Descarga de datos históricos de la compañía deseada
    ticker = yf.Ticker(company)
    hist = ticker.history(start="2015-01-01", end=datetime.now())

    # Escalando los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist["Close"].values.reshape(-1, 1))

    # Creando el conjunto de entrenamiento
    prediction_days = 60
    x_train, y_train = [], []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], prediction_days))

    model = Transformer(input_size=x_train.shape[1], prediction_days=1, dim_feedforward=14).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    epochs = 100
    for epoch in range(epochs):
        inputs = torch.from_numpy(x_train).float().to(device)
        labels = torch.from_numpy(y_train).float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    future_prediction = []
    last_x = scaled_data[-prediction_days:]

    for i in range(days):
        future_input = torch.from_numpy(last_x).float().reshape(1, prediction_days).to(device)
        future_price = model(future_input).to(device)
        future_prediction.append(future_price.detach().cpu().numpy()[0][0])
        last_x = np.append(last_x[1:], future_price.detach().cpu().numpy().reshape(-1, 1))

    prediction = scaler.inverse_transform(np.array(future_prediction).reshape(-1, 1))

    predictions = ["{:.2f}".format(price[0]) for price in prediction]

    return render_template('index.html', predictions=predictions, enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
