# Import libraries
from flask import Flask, render_template, send_file, request, jsonify
import pandas as pd
from io import StringIO
import pickle
import torch
import os
import sys
from mutils import *

# Flask app
app = Flask(__name__)

# data     = pickle.load(open('/Users/kyithinnu/GitHub/NLP/Code/Assignment/A5 - Sentence Embedding with BERT/app/models/data.pkl','rb'))
data     = pickle.load(open('/Users/kyithinnu/GitHub/NLP/Code/Assignment/A5 - Sentence Embedding with BERT/app/models/data.pkl','rb'))
print(data)
word2id  = data['word2id']
print(word2id)
max_len  = data['max_len']
max_mask = data['max_mask']

tokenizer = SimpleTokenizer(word2id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save_path = f'./models/S_BERT.pt'
model = BERT()

# model.load_state_dict(torch.load('app/models/S_BERT.pt', map_location = device))
model.load_state_dict(torch.load('/Users/kyithinnu/GitHub/NLP/Code/Assignment/A5 - Sentence Embedding with BERT/app/models/S_BERT.pt', map_location = device))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    # return the HOME page
    if request.method == 'GET':
        return render_template ('index.html', prompt = '')
    
    if request.method == 'POST':
        # get the user input
        sentence1= request.form.get('sentence1')
        print(sentence1)
        sentence2 = request.form.get('sentence2')
        print(sentence2)
        
        result = calculate_similarity(model, tokenizer, sentence1, sentence2, device)
        print(result)

        return render_template('index.html', result = result, sentence1 = sentence1, sentence2 = sentence2)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
