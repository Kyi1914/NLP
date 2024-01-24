# Importing libraries
from flask import Flask, render_template, request, jsonify
from gensim.models import KeyedVectors
import pickle

# Load the model
model_path = './model/a1-gensim.pkl'
gensim_model = pickle.load(open(model_path,'rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    
    # check the method (GET or POST)
    if request.method == 'GET':
        return render_template('index.html', query='')
    
    if request.method == 'POST':
        
        # get the user input word from the website
        query = request.form.get('query')
        
        # to store the similar words return from the model
        results = []
        
        search = gensim_model.most_similar(query)
        
        for i in range(10):
            results.append(search[i][0])
            
        heading = "Most similar words are:"
        return render_template('index.html', query=query, heading=heading, results=results)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)