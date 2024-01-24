from flask import Flask, render_template, request
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt

# Load the model
model_save_path = './model/a1-gensim.pkl'
gensim = pickle.load(open(model_save_path, 'rb'))

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'GET':
        return render_template('index.html', query='')
    
    if request.method == 'POST':
        query = request.form.get('query')

        # Check if the query is not empty
        if not query:
            error_message = 'Please enter a word.'
            return render_template('index.html', query=query, error_message=error_message)

        # Split the query into individual words
        query_words = query.split()

        results = []  # to store most similar words
        for word in query_words:
            # Check if the word is in the vocabulary
            if word in gensim:
                search = gensim.most_similar(word)
                for i in range(min(10, len(search))):
                    results.append(search[i][0])
            else:
                results.append(f'Word "{word}" not found in vocabulary.')

        # Generate a simple word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(results))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./static/wordcloud.png')

        heading = "Most similar words are:"  # to load only after submitting
        return render_template('index2.html', query=query, heading=heading, results=results)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
