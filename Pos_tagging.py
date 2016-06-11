

# In[116]:

import nltk
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

#Creating the file list for importing sentences from all the input files.
import os
import re
path = "/root/nlp_asg/dependency_treebank"
#file_list = sorted(os.listdir(path), key = lambda x: x)
file_list = os.listdir(path)
fin_file_list = []
for i in file_list:
    file_search = re.search(r'wsj_*',i)
    if file_search:
        fin_file_list.append(i)  

#Read all the files and create a list of list containing tuples where index 0 corresponds to the word and index 1 corresponds
# to the sentence. Every sentence is separated in a new list.
fin = []
new_sent = []
for i in fin_file_list:
    with open(i,'U') as f:
        for line in f:
            line = line.split()
            if len(line) == 0:
                fin.append(new_sent)
                new_sent = []
            else:
                sent_tags = (line[0],line[1])
                new_sent.append(sent_tags)
        fin.append(new_sent)   

#Training the model on UniGram Tagger.
size = int(len(fin) * 0.9)
train_sent = fin[:size]
test_sent = fin[size:]
uni = nltk.UnigramTagger(train_sent)
print uni.evaluate(test_sent)


#Training the model on BiGram Tagger.
bi = nltk.BigramTagger(train_sent)
print bi.evaluate(test_sent)

#Training the model on Hidden Markov Model Tagger using Laplace probability Distribution.
hmm = nltk.tag.hmm.HiddenMarkovModelTrainer()
hmm_model = hmm.train_supervised(train_sent,estimator = nltk.LaplaceProbDist)
print  hmm_model.evaluate(test_sent)

#Training the model on Uni-Bi-Tri backoff Tagger.
default = nltk.DefaultTagger('NN')
uni_b = nltk.UnigramTagger(train_sent,backoff = default)
bi_b = nltk.BigramTagger(train_sent,backoff = uni_b)
tri_b = nltk.TrigramTagger(train_sent,backoff = bi_b)
print tri_b.evaluate(test_sent)

#Creating the list of words (testing set) in each sentence which would be used for tagging.
final  = []
for i in range(len(test_sent)):
    li = []
    for j in test_sent[i]:
        li.append(j[0])
    final.append(li)


#Tagged the testing set according to the model in tag_test function.
def tag_test(model):
    tagged_model = []
    for i in final:
        tagged_model.append(model.tag(i))
    return tagged_model

#Calling the model to tag the test set according to the training model.
uni_tagged = tag_test(uni)
bi_tagged = tag_test(bi)
backoff_tagged = tag_test(tri_b)
hmm_tagged = tag_test(hmm_model)

#Function for calculating the accuracy per sentence.
def sent_accuracy(tagged_set):
    samp = []
    for i in range(len(test_sent)):
        corr_tag_sen = 0
        not_matched_tag = 0
        for j in range(len(test_sent[i])):
            if test_sent[i][j] == tagged_set[i][j]:
                corr_tag_sen += 1
            else:
                not_matched_tag += 1
        tot = corr_tag_sen + not_matched_tag
        samp.append(round(float(corr_tag_sen)/float(tot),4))
    
    return sum(samp)/len(samp)

#Calling the models for the per sentence accuracy.
sent_acc_models = []
sent_acc_models.append(sent_accuracy(uni_tagged))
sent_acc_models.append(sent_accuracy(bi_tagged))
sent_acc_models.append(sent_accuracy(backoff_tagged))
sent_acc_models.append(sent_accuracy(hmm_tagged))


#Calling the models for the overall sentence accuracy.
overall_acc_model = []
overall_acc_model.append(uni.evaluate(test_sent))
overall_acc_model.append(bi.evaluate(test_sent))
overall_acc_model.append(tri_b.evaluate(test_sent))
overall_acc_model.append(hmm_model.evaluate(test_sent))

#Calculating the unique tags in the testing set which would be used for finding the recall and precision.
tags = []
for i in range(len(test_sent)):
    for j in test_sent[i]:
        tags.append(j[1])
unique_tags = list(set(tags))

#Function for calculating the precision and accuracy of the models on the testing set.
def prec_recall(model):
    tags = []
    dict_tags_data = defaultdict(int)
    dict_tags_corr = defaultdict(int)
    dict_tags_op = defaultdict(int)
    #Counting the X's in the data for each tag.
    for tag in unique_tags:
        for i in range(len(test_sent)):
            for j in range(len(test_sent[i])):
                if tag == test_sent[i][j][1]:
                    dict_tags_data[tag] += 1
    #Counting the X's the tagger got right.
                if tag == test_sent[i][j][1] and model[i][j][1] == tag:
                    dict_tags_corr[tag] += 1
    #Counting the X's the tagger output
    for tag in unique_tags:
        for i in range(len(model)):
            for j in range(len(model[i])):
                if tag == model[i][j][1]:
                    dict_tags_op[tag] += 1
    prec =[]
    for i in dict_tags_op:
        if i not in dict_tags_corr.keys():
            prec.append(0)
        else:   
            prec.append(1. *dict_tags_corr[i]/dict_tags_op[i])
    precision =  sum(prec)/len(prec)
    rec =[]
    for i in dict_tags_data:
        if i not in dict_tags_corr.keys():
            rec.append(0)
        else:   
            rec.append(1. *dict_tags_corr[i]/dict_tags_data[i])
    recall =  sum(rec)/len(rec)
    
    return (precision,recall)
    
#Calling the functions to calculate the precision and recall of the models on the test set.
prec_recall_msr = []
prec_recall_msr.append(prec_recall(uni_tagged))
prec_recall_msr.append(prec_recall(bi_tagged))
prec_recall_msr.append(prec_recall(backoff_tagged))
prec_recall_msr.append(prec_recall(hmm_tagged))
precision = [i[0] for i in prec_recall_msr]
recall =  [i[1] for i in prec_recall_msr]

#Graphing the statistics that were calculated for comparing all the models.
def visualisation(statistic,score,title):
    score = [100*i for i in score]
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    bar = ax.bar(ind, score, width = 0.7, color='blue')
    ax.set_ylabel(statistic)
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('UniGram', 'BiGram', 'Backoff', 'HMM'))
    for rect in bar:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    plt.show()

#Visualizing the models on the different statistic which were calculated above.
visualisation('Overall Accuracy Measure',overall_acc_model,'Overall Accuracy')
visualisation('Accuracy Measure Per Sentence',sent_acc_models,'Accuracy Per Sentence')
visualisation('Precision Measure',precision,'Precision')
visualisation('Recall Measure',recall,'Recall')
