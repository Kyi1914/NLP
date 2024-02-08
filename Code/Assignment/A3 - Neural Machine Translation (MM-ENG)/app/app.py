# Import libraries
from flask import Flask, render_template, request, jsonify
from utils import LSTMLanguageModel, generate
import pickle
import torch
import torch.nn.functional as F
from utils import *

# cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Instantiate the model << change the model later>>
# model = initialize_model('multiplicativeAttention')

# web Flask
app = Flask(__name__)

# @app.route('/', method = ['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])

def index():
    # return the HOME page
    if request.method == 'GET':
        return render_template ('index.html', prompt = '')
    
    if request.method == 'POST':
        # get the user input
        prompt = request.form.get('query')
        # print(prompt)
        generation, _ = greedy_decode(model, prompt, max_len=50, device='cpu')
        print(generation)
        sentence = ' '.join(generation)
        return render_template('index.html', prompt = prompt, sentence = sentence)
    
port_number = 8000

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = port_number)
    
