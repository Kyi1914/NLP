# Import libraries
from flask import Flask, render_template, request, jsonify
from utils import LSTMLanguageModel, generate
import pickle
import torch
import torch.nn.functional as F

# cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Import training data
Data = pickle.load(open('./app/model/Data.pkl', 'rb'))
vocab_size   = Data['vocab_size']
emb_dim      = Data['emb_dim']
hid_dim      = Data['hid_dim']
num_layers   =  Data['num_layers']
dropout_rate = Data['dropout_rate']
# lr = 1e-3
tokenizer    = Data['tokenizer']
vocab        = Data['vocab']

# Import the model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('./app/model/fairy-tale-lstm.pt', map_location = torch.device(device)))
model.eval()

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
        prompt = request.form.get('prompt')
        seq_len = 30
        temperature = 0.8
        seed = 0
        generation = generate (prompt, seq_len, temperature, model, tokenizer, vocab, device, seed)
        sentence = ' '.join(generation)
        return render_template('index.html', prompt = prompt, sentence = sentence)
    
port_number = 8000

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = port_number)
    
