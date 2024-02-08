# Import libraries
from flask import Flask, render_template, request, jsonify
# from utils import LSTMLanguageModel, generate
import pickle
import torch
import torch.nn.functional as F
import utils

# cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Instantiate the model << change the model later>>
# model = initialize_model('multiplicativeAttention')
# Instantiate the model
model = utils.define_model()
save_path = f'/Users/kyithinnu/GitHub/NLP/Code/Assignment/A3 - Neural Machine Translation (MM-ENG)/app/models/additiveAttention.pt'
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

# web Flask
app = Flask(__name__)

# @app.route('/', method = ['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])

def index():
    # return the HOME page
    if request.method == 'GET':
        return render_template ('/Users/kyithinnu/GitHub/NLP/Code/Assignment/A3 - Neural Machine Translation (MM-ENG)/app/template/index.html', prompt = '')
    
    if request.method == 'POST':
        # get the user input
        prompt = request.form.get('query')
        # print(prompt)
        generation, _ = utils.greedy_decode(model, prompt, max_len=50, device='cpu')
        print(generation)
        sentence = ' '.join(generation)
        return render_template('/Users/kyithinnu/GitHub/NLP/Code/Assignment/A3 - Neural Machine Translation (MM-ENG)/app/template/index.html', prompt = prompt, sentence = sentence)
    
port_number = 8000

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = port_number)
    
