
import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB

import matplotlib.pyplot as plt
import seaborn as sns


def top_features(train_data, train_labels, vocab, best_params, file_name):
    """Helper function for visualizing the top 5 features per class 
       for a NB model.  Saves png file in images folder."""
    
    # run NB model using best params from model matrix experiment
    classifier = BernoulliNB(**best_params)
    trained_classifier = classifier.fit(train_data, train_labels)
    
    # find the 5 features with largest weights for each class
    weights = np.exp(trained_classifier.feature_log_prob_)
    x_labels = ['inactive', 'active']
    y_indices = [x for i in range(len(x_labels)) for x in (-weights[i]).argsort()[:5]]
    y_labels = [vocab[i] for i in y_indices]
    table = [[weights[i][y] for y in y_indices] for i in range(len(x_labels))]
    
    # convert table to numpy array for vis
    table_array = np.transpose(table)

    # visualize table as a heatmap to spot trends easier
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(table_array, annot=True, cmap='cividis',
                xticklabels=x_labels, yticklabels=y_labels)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=14)
    ax.set_xticklabels(x_labels, fontsize=14)
    plt.title('Top 5 Features Per Class', fontsize='16')
    
    plt.savefig(f'../images/{file_name}.png')
    
    plt.show()
    
    
def top_false_negatives(train_data, train_labels, test_data, test_labels, best_params, n=4):
    """Helper function for finding the indices of the worst false negatives"""
    
    # run NB model using best params from model matrix experiment
    classifier = BernoulliNB(**best_params)
    trained_classifier = classifier.fit(train_data, train_labels)
    
    # generate predicted probabilities and predictions
    probs = trained_classifier.predict_proba(test_data)
    preds = trained_classifier.predict(test_data)

    # find indices and probabilities of all false negatives
    false_neg = np.array([i for i in range(test_data.shape[0]) if (preds[i] == 0 and test_labels[i] == 1)])
    false_neg_probs = np.array([probs[i][0] for i in range(len(false_neg))])
    
    # find indices top 4 false negatives
    top_false_neg = np.array([false_neg[x] for x in (-false_neg_probs).argsort()[:n]])
    
    return top_false_neg


def top_false_positives(train_data, train_labels, test_data, test_labels, best_params, n=4):
    """Helper function for finding the indices of the worst false positives"""
    
    # run NB model using best params from model matrix experiment
    classifier = BernoulliNB(**best_params)
    trained_classifier = classifier.fit(train_data, train_labels)
    
    # generate predicted probabilities and predictions
    probs = trained_classifier.predict_proba(test_data)
    preds = trained_classifier.predict(test_data)

    # find indices and probabilities of all false positives
    false_pos = np.array([i for i in range(test_data.shape[0]) if (preds[i] == 1 and test_labels[i] == 0)])
    false_pos_probs = np.array([probs[i][0] for i in range(len(false_pos))])
    
    # find indices top 4 false negatives
    top_false_pos = np.array([false_pos[x] for x in (-false_pos_probs).argsort()[:n]])
    
    return top_false_pos


def top_negatives(train_data, train_labels, test_data, test_labels, best_params, n=4):
    """Helper function for finding the indices of the most sure negatives"""
    
    # run NB model using best params from model matrix experiment
    classifier = BernoulliNB(**best_params)
    trained_classifier = classifier.fit(train_data, train_labels)
    
    # generate predicted probabilities and predictions
    probs = trained_classifier.predict_proba(test_data)
    preds = trained_classifier.predict(test_data)

    # find indices and probabilities of all false negatives
    neg = np.array([i for i in range(test_data.shape[0]) if (preds[i] == 0 and test_labels[i] == 0)])
    neg_probs = np.array([probs[i][0] for i in range(len(neg))])
    
    # find indices top 4 false negatives
    top_neg = np.array([neg[x] for x in (-neg_probs).argsort()[:n]])
    
    return top_neg


def top_positives(train_data, train_labels, test_data, test_labels, best_params, n=4):
    """Helper function for finding the indices of the most sure positives"""
    
    # run NB model using best params from model matrix experiment
    classifier = BernoulliNB(**best_params)
    trained_classifier = classifier.fit(train_data, train_labels)
    
    # generate predicted probabilities and predictions
    probs = trained_classifier.predict_proba(test_data)
    preds = trained_classifier.predict(test_data)

    # find indices and probabilities of all false negatives
    pos = np.array([i for i in range(test_data.shape[0]) if (preds[i] == 1 and test_labels[i] == 1)])
    pos_probs = np.array([probs[i][0] for i in range(len(pos))])
    
    # find indices top 4 false negatives
    top_pos = np.array([pos[x] for x in (-pos_probs).argsort()[:n]])
    
    return top_pos
