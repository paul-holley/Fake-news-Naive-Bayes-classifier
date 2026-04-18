# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:41:44 2026

@author: paulh
"""


from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# load data
fake = pd.read_csv(r"C:\Users\paulh\OneDrive\Desktop\BAN 675\project\Fake.csv")['text'].to_frame()
fake['label'] = 'fake'
real = pd.read_csv(r"C:\Users\paulh\OneDrive\Desktop\BAN 675\project\True.csv")['text'].to_frame()
real['label'] = 'real'

# merge into one data set with labels
all_news = pd.concat([fake, real], ignore_index = True)


# clean data
all_news['text'] = all_news['text'].str.replace("\W"," ", regex=True)
all_news['text'] = all_news['text'].str.lower()
all_news['text'] = all_news['text'].str.split()

# train test split
np.random.seed(67)
all_news = all_news.sample(frac=1)
split = int(0.8 * len(all_news))


train = all_news.iloc[:split, :].copy()
test  = all_news.iloc[split:, :].copy()




# prep parameter calculation
'''
bad method, tries to fill in 110,000 x 35,000 matrix
wordDist = lil_matrix((len(vocabulary), len(train)))  # ← same shape, fraction of the memory

for wordIdx in range(wordDist.shape[0]):
    for articleIdx in range(wordDist.shape[1]):
        if vocabulary[wordIdx] in train["text"].iloc[articleIdx]:
            wordDist[wordIdx, articleIdx] = 1
            
'''
# convert each article to a count of word frequency
def article_features(word_list):  
    return Counter(word_list)


train_features = [(article_features(train["text"].iloc[i]), train["label"].iloc[i]) 
                  for i in range(len(train))]

test_features  = [(article_features(test["text"].iloc[i]), test["label"].iloc[i]) 
                  for i in range(len(test))]

classifier = NaiveBayesClassifier.train(train_features)
accuracy(classifier, test_features)



# process and get FT, TF articles
# 1. Get predictions
predictions = [classifier.classify(feat) for feat, label in test_features]
true_labels = [label for feat, label in test_features]

# 2. Confusion matrix
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.labels())
disp.plot()

# 3. Find misclassified articles
test = test.reset_index(drop=True)

false_positives = []  # predicted fake, actually real
false_negatives = []  # predicted real, actually fake

for i, (pred, true) in enumerate(zip(predictions, true_labels)):
    if pred == "fake" and true == "real":
        false_positives.append(i)
    elif pred == "real" and true == "fake":
        false_negatives.append(i)
        


fp_articles = test.iloc[false_positives]
fn_articles = test.iloc[false_negatives]

fp_articles = fp_articles.copy()
fn_articles = fn_articles.copy()

fp_articles["text"] = fp_articles["text"].str.join(" ")
fn_articles["text"] = fn_articles["text"].str.join(" ")

fp_articles.to_csv("false_positives.csv", index=False)
fn_articles.to_csv("false_negatives.csv", index=False)

print(f"False positives : {len(fp_articles)}")
print(f"False negatives : {len(fn_articles)}")