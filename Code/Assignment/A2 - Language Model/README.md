# A2: Language Model

This assignment is completed under the guidline of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
- [Overview of this assignment](#overview-of-this-assignment)
    - [Brief Introduction](#brief-introduction)
    - [ Task1. Dataset Acquision](#task-1-dataset-acquision)
    - [Task 2. Model Training](#task-2-model-training)
        - [2.1 Data Preprocessing](#21-data-preprocessing)
        - [2.2 Model Architecture](#22-model-architecture)
        - [2.3 Training Process](#23-training-process)
    - [Task 3. Text Generation - Web Application Development](#task-3-text-generation---web-application-development)


## Overview of this assignment

### Brief Introduction
In this assignment, I will focus on building a language model using a text dataset of (). The objective is to train a model that can generate coherent and contextually relevant text based on a given input. Additionally, I will develop a simple web application to demonstrate the capabilities of my language model interactively.

### Task 1. Dataset Acquision

For my language modeling project, I curated a dataset focused on fairy tales. The primary source for this dataset was Project Gutenberg's collection of fairy tales, which provides a diverse selection of timeless stories. These tales were obtained in the form of text files from the Project Gutenberg website (https://www.gutenberg.org/ebooks).

<b> Dataset Details: </b>
- Dataset Name: KyiThinNu/FairyTales
- Source: Project Gutenberg's collection of fairy tales
- Dataset Content: Text files containing a variety of fairy tales
- Collection Process: The text files were downloaded from Project Gutenberg, ensuring a representative mix of fairy tales from different cultures and authors.
- Dataset Location on Hugging Face: KyiThinNu/FairyTales on Hugging Face

<b> Dataset Split:</b>
I performed a standard split to create distinct sets for training, testing, and validation, ensuring that the model is trained on a diverse range of fairy tales and tested on unseen data.

To make this dataset accessible to the wider community and facilitate collaboration, I have uploaded it to the Hugging Face Model Hub. Researchers and practitioners interested in fairy tale language modeling can easily access and utilize the dataset for their projects.

### Task 2. Model Training
#### 2.1 Data Preprocessing
My original dataset is text file. In order to get the formatted data for my model, I do the Tokenization and Numericalization.
let's consider my data from my dataset is "I like reading"

    - Tokenization
    To tokenized my text data, I use the basic_english tokenizer, a simple tokenizer provided by torchtext that tokenizes text into basic English words while ignoring punctuation and other non-alphabetic characters. This tokenizer is often used as a starting point for natural language processing tasks when the goal is to process English text at a word level.
    After this step I will get ["I", "like", "reading"]

    - Numericalization
    Then I construct a vocabulary from the tokenized training dataset with a minimum frequency threshold.
    It includes the most frequent words from the training set, with a minimum frequency of 3, and introduces special tokens '<unk>' and '<eos>'. The default index for unknown tokens is set to the index of '<unk>'. '<eos>' will be used to train the model as an ending the sentence. This vocabulary will be useful for converting words into numerical indices for further use in neural network models.

    - Prepare the batch loader
    <get_data> function is to perform data preparation for a sequence-to-sequence model by numericalizing the tokens using a vocabulary (vocab).
        - Adding '<eos>' Token: 
            The '<eos>' token is added at the end of each sentence within the dataset.
        - Numericalization:
            The tokens in each example are numericalized using the provided vocabulary (vocab). Each token is replaced with its corresponding index in the vocabulary.
        - Data Preparation for Embedding:
            The resulting numericalized tokens are concatenated into the data list.
        - Tensor Conversion:
            The data list is converted to a PyTorch LongTensor. 
        - Batching:
            The function calculates the number of batches based on the specified batch_size.
            The data tensor is trimmed to ensure all batches are even.
            Finally, the tensor is reshaped to have dimensions [batch_size, seq_len].

The resulting data tensor can be used as input for embedding layers in a sequence-to-sequence model. Each row represents a batch of sequences, and each element in a row represents the index of a word in the vocabulary.

#### 2.2 Model Architecture
Next, defines a Long Short-Term Memory (LSTM) language model in PyTorch.
This model is designed for language modeling tasks where the goal is to predict the next word in a sequence given the previous words. The LSTM architecture allows capturing sequential dependencies in the data.

LSTMLanguageModel architecture includes these key components:
1. Embedding Layer : to embed the incoming word [batch-size, seq_len, emb_dim]
2. LSTM Layer : stacked LSTM (2 layer)
3. Dropout Layer: use embedding dropout and variation dropout regularization 
4. Fully Connected(linear) Layer : update the dimension [batch_size, seq_len, vocab_size]
5. Initializing of weights: uniform sampling within predefined range
6. Initializing of hidden states: initialize with zeros for every epoch
7. Detaching the hidden states: remove from computational graph
7. Forward Pass

In summary, the model utilizes an LSTM architecture to capture sequential dependencies in the input sequences, and the fully connected layer generates predictions for each word in the vocabulary. Dropout is applied to enhance generalization, and appropriate weight initialization is performed for stable training.

#### 2.3 Training Process
Training process is as follows:

Initialized Parameters:

vocab_size   = len(vocab)  
emb_dim      = 1024 # 400 in the paper  
hid_dim      = 1024 # 1150 in the paper  
num_layers   = 2 # 3 in the paper  
dropout_rate = 0.65  
lr           = 1e-3  

epoch                = 50  
trainable parameters = 43,328,150  

1. set the drop out layer to training mode.
2. reshape the input dimensions
3. initialize hidden layers (reset after every epoch)
4. get_batch: which return the source and target batches (which are equal length = 50)
5. forward pass and back propagate
6. limit the exploding gradient (0.25) using <`clip_grad_norm`> function
7. during training, use <`ReduceLROnPlateau`> function to schedule learning rate
8. only save model with the best validation loss.

### Task 3. Text Generation - Web Application Development

Web Application serve as an interface between the model and users.
- Take the input from the user as a text from the text box.
- The generate function will process user input with the model.
- Show the result as a text on the web page again.
    
The generate function process flow is as follows:
- Take the user input and predict next word
- Add next word to input
- Predict another word using the new input (user input + predicted words)
- Repeat until max sequence length or '<`eos`>' token.
- Join elements in output list and return to the website to show the user.