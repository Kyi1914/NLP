from flask import Flask, render_template, request
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt

# Load the model
model_skipgram    = './model/A1-Skipgram.pt'
model_NegSampling = './model/A1-NegSampling.pt'
model_glove       = './model/A1-Glove.pt'
model_gensim      = './model/a1-gensim.pkl'

gensim = pickle.load(open(model_gensim, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'GET':
        return render_template('index.html', query='')
    
    if request.method == 'POST':
        query = request.form.get('query')

        # Check if the query is not empty
        if not query:
            error_message = 'Please enter a word.'
            return render_template('index.html', query = query, error_message = error_message)

        # Split the query into individual words
        query_words = query.split()

        results       = []  # to store most similar words
        vectors       = [] # to store all input vectors
        result_vector = [] # to store final result
        
        for word in query_words:
            
            # Check if the word is in the vocabulary
            if word in gensim:
                vectors.append(gensim.get_vector(word))
            else:
                results.append(f'Word "{word}" not found in vocabulary.')

        for i in range(len(vectors)):
            if i == 0:
                result_vector = vectors[i]
                print(result_vector)
            else:
                result_vector = result_vector + vectors[i] # combine the embeddings of all input words
        print(result_vector)
        
        search = gensim.most_similar(result_vector)
        
        for i in range(len(search)):
            results.append(search[i][0])

        # Generate a simple word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(results))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./static/wordcloud.png')

        heading = "Most similar words are:"  # to load only after submitting
        return render_template('index.html', query=query, heading=heading, results=results)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
