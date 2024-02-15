import numpy as np 
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_md')

# define path
entity_path = '/Users/kyithinnu/GitHub/NLP/Code/Assignment/A4 - Resume Parser/app/data/entity.jsonl'

# Add an EntityRuler for skills, educationa and certification
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.from_disk(entity_path)

# define pattern
patterns = [
    {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": "((\d){7})"}}]},
    {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\\b"}}]}
]
# add pattern to ruler
ruler.add_patterns(patterns)

# -----------------------------------------------------------------------------

# Text Clenaing

# remove hyperlinks

import re

def remove_hyperlinks(sentence):
    
    #just in case there is hyperlink....
    sentence = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        sentence
    )
    
    return sentence

# clean the data
from spacy.lang.en.stop_words import STOP_WORDS

def preprocessing(sentence):
    
    # remove hyperlink
    sentence = remove_hyperlinks(sentence)
    
    stopwords    = list(STOP_WORDS)
    doc          = nlp(sentence)
    clean_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and \
            token.pos_ != 'SYM' and token.pos_ != 'SPACE':
            clean_tokens.append(token.lemma_.lower().strip())
    
    return " ".join(clean_tokens)

# -----------------------------------------------------------------------
# Extract Information

# get the unique skills
def unique_skills(x):
    return list(set(x))

from collections import defaultdict

# ----------------------------------------------

from PyPDF2 import PdfReader

def readPDF(path):
    reader = PdfReader(path)
    
    page = reader.pages[0]
    text = page.extract_text()
    text = preprocessing(text)
    doc = nlp(text)

    # person    = []
    email     = []
    phone     = []
    skill     = []
    education = []
    organization = []

    for ent in doc.ents:
        if ent.label_ == 'PHONE':
            phone.append(ent.text)
        if ent.label_ == 'EMAIL':
            email.append(ent.text)
        if ent.label_ == 'SKILL':
            skill.append(ent.text)
        if ent.label_ == 'EDUCATION':
            education.append(ent.text)
        if ent.label_ == 'ORG':
            organization.append(ent.text)
    
    phone     = set(phone)  
    email     = set(email)     
    skill     = set(skill)
    education = set(education)
    organization = set(organization)
    
    info = {'phone': phone, 'email': email, 'skills':skill,'education':education, 'organization': organization}
    df = pd.DataFrame.from_dict(info, orient='index') # for dataframe
    
    return info, df



