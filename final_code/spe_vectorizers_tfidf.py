
import pandas as pd
import numpy as np

import codecs
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer


def spe_featurizer_tfidf(train_data, test_data):
    """Creates datasets ready to input into ML models
       Uses SPE vocabulary and CountVectorizer to create features"""
    
    # load vocab
    spe_vob = codecs.open(r'SPE_ChEMBL.txt')
    spe = SPE_Tokenizer(spe_vob)
    
    # split SMILES strings to tokens
    train_spe = train_data.apply(lambda x: spe.tokenize(x))
    test_spe = test_data.apply(lambda x: spe.tokenize(x))
    
    # split tokenized strings into tokens
    # transform dataset into a matrix of vectors
    split_string = lambda x: x.split()
    vectorizer = TfidfVectorizer(preprocessor=None, stop_words=None, lowercase=False, tokenizer=split_string)

    x_train = vectorizer.fit_transform(train_spe)
    x_test = vectorizer.transform(test_spe)
    train_vocab = vectorizer.get_feature_names()
    
    return x_train, x_test, train_vocab


def spe_featurizer_tfidf2(train_data, test_data):
    """Creates datasets ready to input into ML models
       Uses SPE vocabulary and CountVectorizer to create features
       Forces the use of whole vocabulary, not just fragments in train data"""
    
    # load vocab
    spe_vocab = pd.read_csv('SPE_ChEMBL.txt', header=None)
    spe_vocab = spe_vocab.rename(columns={0: 'fragments'})
    
    spe_vob = codecs.open(r'SPE_ChEMBL.txt')
    spe = SPE_Tokenizer(spe_vob)
    
    # split SMILES strings to tokens
    train_spe = train_data.apply(lambda x: spe.tokenize(x))
    test_spe = test_data.apply(lambda x: spe.tokenize(x))
    
    # split tokenized strings into tokens
    # transform dataset into a matrix of vectors
    split_string = lambda x: x.split()
    vectorizer = TfidfVectorizer(preprocessor=None, stop_words=None, lowercase=False, 
                                 tokenizer=split_string, vocabulary=spe_vocab.fragments)
    x_train = vectorizer.transform(train_spe)
    x_test = vectorizer.transform(test_spe)
    train_vocab = vectorizer.get_feature_names()
    
    return x_train, x_test, train_vocab


def atom_featurizer_tfidf(train_data, test_data):
    """Creates datasets ready to input into ML models
       Uses atomwise tokenizer and CountVectorizer to create features"""
    
    # split SMILES strings into tokens
    train_atom = train_data.apply(lambda x: ' '.join(atomwise_tokenizer(x)))
    test_atom = test_data.apply(lambda x: ' '.join(atomwise_tokenizer(x)))
    
    split_string = lambda x: x.split()
    vectorizer = TfidfVectorizer(preprocessor=None, stop_words=None, 
                                 lowercase=False, tokenizer=split_string)

    x_train = vectorizer.fit_transform(train_atom)
    x_test = vectorizer.transform(test_atom)
    train_vocab = vectorizer.get_feature_names()
    
    return x_train, x_test, train_vocab


def kmer_featurizer_tfidf(train_data, test_data):
    """Creates datasets ready to input into ML models
       Uses atomwise tokenizer a
       nd CountVectorizer to create features"""
    
    # split SMILES strings into tokens
    train_kmer = train_data.apply(lambda x: ' '.join(kmer_tokenizer(x)))
    test_kmer = test_data.apply(lambda x: ' '.join(kmer_tokenizer(x)))
    
    split_string = lambda x: x.split()
    vectorizer = TfidfVectorizer(preprocessor=None, stop_words=None, 
                                 lowercase=False, tokenizer=split_string)

    x_train = vectorizer.fit_transform(train_kmer)
    x_test = vectorizer.transform(test_kmer)
    train_vocab = vectorizer.get_feature_names()
    
    return x_train, x_test, train_vocab
