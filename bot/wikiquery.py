# coding: utf-8

import io
import sys
import random
import string # to process standard python strings
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

def process_wiki(word):
    import wikipedia
    try:
        p = wikipedia.page(word)
    except wikipedia.DisambiguationError as e:
        s = random.choice(e.options)
        s = e.options[0]
        p = wikipedia.page(s)

    raw = p.content.lower()
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    
    return sent_tokens, word_tokens

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(topic, user_response):
    sent_tokens, _ = process_wiki(topic)
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        return "Sorry, no answer"
    else:
        return sent_tokens[idx]


def query(topic, user_response):
    user_response = user_response.lower()
    res = ""
    print(topic)
    print(user_response)


    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            res = "You are welcome.."
        else:
            if(greeting(user_response)!=None):
                res = greeting(user_response)
            else:
                res = response(topic, user_response)
    else:
        res = "Bye! take care.."

    return res
