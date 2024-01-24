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

## Report
 Model             | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|------------------|-------------|---------------|---------------|--------------------|------------|
| Skipgram         | 2           | 30.60         | 31 min 35 sec | 0.00%              | 0.00%     |
| Skipgram (NEG)   | 2           | 10.55         | 31 min 51 sec | 0.00%              | 0.00%     |
| Glove            | 2           | 6.93          | 4 min 12 sec  | 0.00%              | 0.00%     |
| Glove (Gensim)   | -           | -             | -             | 55.45%             | 93.87%    |