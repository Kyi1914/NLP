# A2: Language Model

This assignment is completed under the guidline of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
[Overview of this assignment]

## Overview of this assignment

### Brief Introduction
In this assignment, I will focus on building a language model using a text dataset of (). The objective is to train a model that can generate coherent and contextually relevant text based on a given input. Additionally, I will develop a simple web application to demonstrate the capabilities of my language model interactively.

### Task 1. Dataset Acquision

For my language modeling project, I curated a dataset focused on fairy tales. The primary source for this dataset was Project Gutenberg's collection of fairy tales, which provides a diverse selection of timeless stories. These tales were obtained in the form of text files from the Project Gutenberg website (https://www.gutenberg.org/ebooks).

<b> Dataset Details: </b>
- Dataset Name: Kyi/FairyTales
- Source: Project Gutenberg's collection of fairy tales
- Dataset Content: Text files containing a variety of fairy tales
- Collection Process: The text files were downloaded from Project Gutenberg, ensuring a representative mix of fairy tales from different cultures and authors.
- Dataset Location on Hugging Face: Kyi/FairyTales on Hugging Face

<b> Dataset Split:</b>
I performed a standard split to create distinct sets for training, testing, and validation, ensuring that the model is trained on a diverse range of fairy tales and tested on unseen data.

To make this dataset accessible to the wider community and facilitate collaboration, I have uploaded it to the Hugging Face Model Hub. Researchers and practitioners interested in fairy tale language modeling can easily access and utilize the dataset for their projects.

### Task 2. Model Training
- 2.1 Data Preprocessing
My original dataset is text file. In order to get the formatted data for my model, I do the Tokenization and Numericalization.
let's consider my data from my dataset is "I like reading"

    - Tokenization
    To tokenized my text data, I use the basic_english tokenizer, a simple tokenizer provided by torchtext that tokenizes text into basic English words while ignoring punctuation and other non-alphabetic characters. This tokenizer is often used as a starting point for natural language processing tasks when the goal is to process English text at a word level.
    After this step I will get ["I", "like", "reading"]

    - Numericalization
    Then I construct a vocabulary from the tokenized training dataset with a minimum frequency threshold.


- 2.2 Model Architecture

- 2.3 Training Process


### Task 3. Text Generation - Web Application Development
