#!/usr/bin/env python
# coding: utf-8
import pickle

import fitz
import numpy as np
import pandas as pd
import os
import re
import spacy
import torch
from nltk.tokenize import sent_tokenize
import sys
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


nltk.download('punkt')

user_id = sys.argv[-1]


def extract_text(filename):
    """
    Extracts the text from the input file.
    :param filename: path to the file
    :return: the extracted raw text
    """
    try:
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        return None


# Reads all files in the uploaded files folder of the user to a Dataframe
path_to_pdfs = './uploaded_files/pdf/' + user_id + '/'

df = pd.DataFrame()
df['filename'] = os.listdir(path_to_pdfs)
df['filename'] = path_to_pdfs + df['filename']
df['text'] = df['filename'].apply(extract_text)
df = df[df['text'].notna()]

if df.shape[0] < 1:
    raise Exception('Empty Dataframe')

# drop duplicates
df['text_low'] = df['text'].apply(lambda text: ''.join([t.lower() for t in text]))
df = df.drop_duplicates(subset='text_low')  # drop articles with the same abstract
df = df.drop(columns=['text_low'])

# remove special characters, double spaces, \n
df['text'] = df['text'].apply(lambda text: text.replace('\n', ''))
df['text'] = df['text'].apply(lambda text: re.sub(' +', ' ', str(text)))  # replace multiple spaces
df['text'] = df['text'].apply(lambda text: re.sub('\. +', '.', str(text)))  # replace multiple periods
df['text'] = df['text'].apply(lambda text: re.sub('\.+', '. ', str(text)))  # replace multiple periods

# sort out everything after (c) or ©
df['text'] = df['text'].apply(lambda text: ''.join(text.split('©')[:-1]) if '©' in text else text)
remove = df['text'].apply(lambda text: '(c)' + text.split('(c)')[-1] if '(c)' in text else text).value_counts()
remove = list(remove[remove > 1].index)
remove2 = df['text'].apply(lambda text: '(C)' + text.split('(C)')[-1] if '(C)' in text else text).value_counts()
remove = remove + list(remove2[remove2 > 1].index)
regex = re.compile('|'.join(map(re.escape, remove)))
df['text'] = df['text'].apply(lambda text: regex.sub('', text))

# drop very short texts
df = df[df['text'].apply(len) > 100]

if df.shape[0] < 1:
    raise Exception('Empty Dataframe')

# split into sentences
nlp = spacy.load("en_core_web_sm")
df['sents'] = df['text'].apply(sent_tokenize)
df = df[df['sents'].apply(len) > 0]
df_sent = df.explode('sents')
df_sent = df_sent.drop_duplicates(subset=['sents'])


def create_n_sent_windows(in_df, window_size):
    """
    Splits the text into text segments of n sentences with overlapping sentences.
    :param in_df: input dataframe
    :param window_size: number of sentences per segment
    :return: dataframe of text segments
    """
    in_df['merged'] = in_df['sents'].apply(
        lambda sents: [(s, r.values) for (s, r) in zip(sents, pd.Series(sents).rolling(window_size, center=True))])
    in_df = in_df.explode('merged')
    in_df['sents'] = in_df['merged'].apply(lambda x: x[0])
    in_df['windowed_' + str(window_size)] = in_df['merged'].apply(lambda x: ' '.join(x[1]))
    in_df = in_df[['sents', 'windowed_' + str(window_size)]].drop_duplicates()
    return in_df


# splits the text into segments of 3 sentences
df_3 = create_n_sent_windows(df, 3)
df = df_sent.merge(df_3, on='sents', how='left').drop_duplicates(subset='sents').drop(columns=['text'])

# predict topic/class for each segment
tokenizer = AutoTokenizer.from_pretrained("model-checkpoint")

id2label = pickle.load(open('id2label.pckl', 'rb'))
label2id = pickle.load(open('label2id.pckl', 'rb'))

model = AutoModelForSequenceClassification.from_pretrained("model-checkpoint", num_labels=len(label2id.keys()),
                                                           id2label=id2label, label2id=label2id)


def predict(text, thresh=0.3):
    """
    Predicts the topic/class for a text segment if the probability is higher than the threshold.
    :param text: input text
    :param thresh: threshold for prediction
    :return: topic/class
    """
    encoding = tokenizer(text, return_tensors="pt")
    outputs = model(**encoding).logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    if np.max(predictions) > thresh:
        return id2label[predictions[np.argmax(probs)][0]]
    else:
        return None


df['topic'] = df['windowed_3'].apply(predict)
df = df[df['topic'].notna()]

if df.shape[0] < 1:
    raise Exception('Empty Dataframe')

df = df.rename(columns={'windowed_3': 'text'})
# save preprocessed dataframe
df[['text', 'topic']].to_pickle('./Application/uploaded_files/df_uploaded_' + user_id + '.pckl')
