from flask import Flask, render_template, request
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import os
import torch
import numpy as np  # added numpy

app = Flask(__name__)

# Paths to the models
model_skipgram = './model/A1-Skipgram.pt'
model_NegSampling = './model/A1-NegSampling.pt'
model_glove = './model/A1-Glove.pt'
model_gensim = './model/a1-gensim.pkl'

# Load the gensim model
gensim = pickle.load(open(model_gensim, 'rb'))

# Load other models
models = {
    'skipgram': torch.load(model_skipgram),
    'negsampling': torch.load(model_NegSampling),
    'glove': torch.load(model_glove)
}

# load the data
Data = pickle.load(open('./model/Data.pkl', 'rb'))
corpus = Data['corpus']
vocab = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['embedding_size']

# cosine_similarity 
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# calculate_similarity of two vectors
def calculate_similarity_numpy(word_vectors, result_vector):
    
    # Calculate cosine similarities using numpy
    similarities = [cosine_similarity(result_vector, word_vector) for word_vector in word_vectors]

    # Find the index of the word with the highest similarity
    closest_word_index = np.argmax(similarities)

    return closest_word_index

# route
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'GET':
        return render_template('index.html', query='')

    if request.method == 'POST':
        query = request.form.get('query')

        if not query:
            error_message = 'Please enter a word.'
            return render_template('index.html', query=query, error_message=error_message)

        # if enter 2 words
        query_words = query.split() # ["ice", "cream"]

        results = {model: [] for model in models.keys()}
        # {'skipgram': [], 'negsampling': [], 'glove': []}
        
        # Get word vectors for all words in the vocabulary
        all_word_vectors = torch.stack([model.get_embed(word) for word in vocab])

        for model_name, model in models.items():

            # loop for each word for user_passed word
            for word in query_words:

                # Check if the word is in the vocabulary
                emb_word = [model.get_embed(word) if word in vocab else model.get_embed('<UNK>')]

                # find similar word
                if word in vocab:
                    # Convert PyTorch tensors to NumPy arrays
                    all_word_vectors_np = all_word_vectors.numpy()
                    emb_word_np = emb_word[0].numpy()

                    # Calculate similarity using numpy
                    closest_word_index_np = calculate_similarity_numpy(all_word_vectors_np, emb_word_np)

                    # Get the closest word from your vocabulary
                    closest_word_np = vocab[closest_word_index_np]

                    results[model_name].append((word, closest_word_np))

        # Generate a simple word cloud for gensim
        gensim_results = []  # to store most similar words for gensim
        for word in query_words:
            if word in gensim:
                gensim_results.extend(gensim.most_similar(word))

        results['gensim'] = gensim_results

        # Generate word clouds for all models
        # for model_name, model_results in results.items():
        #     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([w[0] for w in model_results]))
        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(wordcloud, interpolation='bilinear')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join('./static', f'{model_name}_wordcloud.png'))

        return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
