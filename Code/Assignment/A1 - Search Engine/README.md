# A1: Search Engine

This is the assignment:
- for [2024-AIT] : Artificial Intelligence: Natural Language Understanding
- by st124087 (Kyi Thin Nu)

## Brief Introduction

This assignment focuses on creating a search engine, a practical application of information retrieval in natural language processing. The search engine, deployed on a website, should return the top paragraphs with the highest similarity to a given query, such as "Harry Potter". 

Objectives:
 - to understand and implement word embedding techniques, and
 - creating a web interface for the search engine

## Credit to the dataset sourch that I used in this project
The "rural.txt" dataset used in this project is part of the Natural Language Toolkit (NLTK) datasets, which is available at https://www.nltk.org/nltk_data/. The dataset is provided by NLTK for educational and research purposes. We acknowledge and appreciate the contributions of the NLTK project to natural language processing research and education.

## Project Workflow
 - Task (1) : Model and Training
    I created Word2Vec models, including Skip-gram with and without negative sampling, GloVe, and Gensim. These models were trained using the provided [dataset](#credit-to-the-dataset-sourch-that-i-used-in-this-project) that I mentioned the above.

- Task (2) : Model Comparison and Analysis
    I conducted a comprehensive comparison and analysis of the models. This involved:
    - Evaluating Skip-gram, Skip-gram with negative sampling, and GloVe models based on training loss and training time.
    - Assessing model performance using a Word Analogies dataset, calculating both syntactic and semantic accuracy.

    Detailed results of the model comparison can be found in [the Model Comparison and Analysis section](#model-comparison-and-analysis).

- Task (3) Search Engine - Web Application Development
    - I developed a simple and user-friendly web application for the search engine. The application features an input text box for users to submit search queries.

## Model Comparison and Analysis
(1) Compare Skip-gram, Skip-gram negative sampling, GloVe models on training loss, training time.
(2) Using Word analogies dataset and calucalte between syntactic and semantic accuracy, 

 Model             | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|------------------|-------------|---------------|---------------|--------------------|------------|
| Skipgram         | 2           | 7.9912         | 3 min 8 sec | 0.00%              | 0.00%     |
| Skipgram (NEG)   | 2           | 0.7983         | 3 min 1 sec | 0.00%              | 0.00%     |
| Glove            | 2           | 6.9831         | 1 min 0 sec | 0.00%              | 0.00%     |
| Glove (Gensim)   | -           | -              | -           | 55.45%             | 93.87%    |

(3)The Similarity Metrics (Correlation between my models and human judgement)

Model             | Skipgram | NEG | GloVe | GloVe (gensim) | Y-true |
|----------------- |---------|-----|-------|----------------|--------|
| Spearmanr Result | 0.0939  | 0.0844 | 0.0845  | -0.6035   |   |