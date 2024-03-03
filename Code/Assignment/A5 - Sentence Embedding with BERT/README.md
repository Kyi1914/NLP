# A5: Sentence Embedding with BERT

This assignment is completed under the guidance of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
- [Overview of this assignment](#overview-of-this-assignment)
- [ Task1: Training BERT from Scratch with Sentence Transformer ](#task-1-training-bert-from-scratch-with-sentence-transformer)
- [ Task2: Sentence Embedding with Sentence BERT](#task-2-sentence-embedding-with-sentence-bert)
- [ Task3: Evaluation and Analysis](#task-3-evaluation-and-analysis)
- [ Task4: Text similarity - Web Application Development](#task-4-text-similarity---web-application-development)
- [Testing Purpose](#testing)

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

The notebook "comparisonWithPretrain.ipynb" presents a comparative analysis between a scratch-trained model and pretrained models from Hugging Face. The objective is to evaluate performance, robustness, and training efficiency, providing insights into the strengths and weaknesses of each approach.

I tested with the two sentences for the two model (scratch and pretrained).
- sentence_a = 'Your contribution helped make it possible for us to provide our students with a quality education.'
- sentence_b = "Your contributions were of no help with our students' education."

<table>
    <tr>
        <th>Model</th>
        <th>Cosine Similarity Score</th>
    </tr>
    <tr>
        <th>S BERT (pretrained) </td>
        <td> 0.38588038 </td>
    </tr>
    <tr>
        <th>S BERT (scratch) </td>
        <td> 0.9748 </td>
    </tr>
    <tr></tr>
</table> 

My model's suboptimal performance can be attributed to several factors, including a limited dataset, minimal epochs, and constraints on hyperparameters during training. This report outlines the identified issues and proposes specific strategies for enhancing the model's effectiveness.

<h4> Issues Identified: </h4>

- Limited Dataset: The model was trained on a dataset of insufficient size, limiting its exposure to diverse patterns and scenarios.

- Minimal Epochs: Insufficient training epochs hindered the model's ability to converge and capture intricate relationships within the data.

- Limited Hyperparameter Tuning: The hyperparameters were not extensively tuned, preventing the model from reaching its optimal configuration.

<h4> Recommendations for Improvement: </h4>

- Increase Epochs:

    -   Gradually increase the number of training epochs to allow the model to capture more nuanced patterns.
    - Monitor performance metrics to identify the point of diminishing returns and avoid overfitting. 

- Hyperparameter Tuning:

    - Conduct a thorough exploration of hyperparameter configurations.
    - Utilize techniques like grid search or random search to identify optimal hyperparameter values.
    - Focus on parameters such as learning rates, batch sizes, and regularization terms.

- Train with Larger Dataset:

    - Expand the training dataset to provide the model with a more diverse set of examples.
    - A larger dataset enhances the model's ability to generalize and improves performance on real-world data.


## Task 4: Text similarity - Web Application Development
I develop a simple web application to demonstrate text similarity using a text-embedding model. 
Developing web application includes backend development, model integration, and frontend development. For creating such an application I use Flask for the backend and basic HTML/JavaScript for the frontend.

The HTML file will contain a simple form for inputting two sentences and a button to submit them.

When the submit, cosine similarity of the two sentences is calculated and show the result.

## Testing Purpose

Please run the app.py to start the web application.

## Contributors
Special thanks to Ma Wut Yee Aung, Minn Banya and Rakshya for helping me with this assignment.