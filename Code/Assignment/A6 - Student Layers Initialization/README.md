# A6: Student Layers Initialization

This assignment is completed under the guidance of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
- [Overview of this assignment](#overview-of-this-assignment)
- [ Task1:  Student Layer Initialization ](#)
- [ Task2: Evaluation and Analysis](#)

## Overview of this assignment

### Brief Introduction
In this assignment, I will explore Student Layers Initialization via distillation using BERT Huggingface.

## Task 1: Training BERT from Scratch with Sentence Transformer
Based on case-studies/distilBERT.ipynb, I modify the followings:
1) Initialize the top K layers    {1,2,3,4,5,6}    from 12-layers teacher to 6-layers student.
2) Initialize the bottom K layers {7,8,9,10,11,12} from 12-layers teacher to 6-layers student.
3) Initialize the odd layers      {1,3,5,7,8,9,11} from 12-layers teacher to 6-layers student. 

In the case-studies implement by initializing with the even layers {2,4,6,8,10,12} for 6-layers student.

## Task 2: Sentence Embedding with Sentence BERT
1) I perform a detailed evaluation of my distilled student model, analyzing the impact of the initial layer selection (top K layers, bottom K layers, odd layers) on its performance.
2) I train the model using mnli task from GLUE dataset using the follwings randomized data size.
    - the training data: 1000
    - validation_mismatched: 50
    - test_mismatched: 50
    - hyperparameters:
        - batch_size         : 32
        - Teacher parameters : 109484547
        - Student parameters : 66957315
        - lr                 : 5e-5
        - num_epochs         : 5

 Overall performance for each model is as follows :

<table>
    <tr>
        <th>Student Layer</th>
        <th>Training Loss</th>
        <th>Validation Loss</th>
        <th>Validation Accuracy</th>
    </tr>
    <tr>
        <th>Top K Layer </td>
        <td> 0.38747263718396424 </td>
        <td> 1.0636069178581238 </td>
        <td> 0.38 </td>
    </tr>
    <tr>
        <th>Bottom K Layer </td>
        <td> 0.3865556204691529 </td>
        <td> 1.0636069178581238 </td>
        <td> 0.38 </td>
    </tr>
    <tr>
        <th>Odd Layer </td>
        <td> 0.3876251710578799 </td>
        <td> 1.0636069178581238 </td>
        <td> 0.38 </td>
    </tr>
    <tr>
        <th>Even Layer </td>
        <td> 0.4032769039273262 </td>
        <td> 1.0733229994773865 </td>
        <td> 0.38 </td>
    </tr>
    <tr></tr>
</table> 

## For Preview
for the detailed implementation and performances, please visit to my A6.ipynb.

Thank you for your interesting.