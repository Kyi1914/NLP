# A5: Sentence Embedding with BERT

This assignment is completed under the guidance of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
- [Overview of this assignment](#overview-of-this-assignment)
- [ Task1: Implementation Foundation ](#task-1-implementation-foundation)
- [Task2: Resume Parsing Features - Web Application Development](#task-2-resume-parsing-features---web-application-development)

## Overview of this assignment

### Brief Introduction
In this assignment, I will emphasis on leveraging text embeddings and capturing semantic similarity using a powerful encoder like BERT.

## Task 1: Training BERT from Scratch with Sentence Transformer
I extend the BERT.ipynb, which is based on masked language model.  
- implement Bidirectional Encoder Representation from Transformer (BERT) from scratch.
- I train the model using the IMDB Dataset from kaggle. 
the dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- then I save the trained model weights as "data.pkl" for later use in Task 2.

## Task 2: Sentence Embedding with Sentence BERT
I implement pretrained BERT network that use siamese network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.


## Task 3: Evaluation and Analysis
- (1)
my model

- (2)
compare to others models

- (3)
why my model is bad
because 

how should I improve the model
more epoch, more hyperparmeter tuning, train with larger dataset

## Task 4: Text similarity - Web Application Development
I develop a simple web application to demonstrate text similarity using a text-embedding model. 
Developing web application includes backend development, model integration, and frontend development. For creating such an application I use Flask for the backend and basic HTML/JavaScript for the frontend.

This HTML file will contain a simple form for inputting two sentences and a button to submit them.


![Alt Text](./app/image/ui1.png)  

1. User can upload a PDF.  

![Alt Text](./app/image/ui2.png)  


2. Then system will extract the information including contact, skills, education and worked organization from the user uploaded file.  

![Alt Text](./app/image/ui3.png)  


3. The system will support to download the csv file version including the extracted information provided from the system.  

![Alt Text](./app/image/ui4.png) 