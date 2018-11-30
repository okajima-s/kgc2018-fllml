#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import sys
import json
import argparse

import numpy as np
import scipy.spatial.distance as dis

from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

# -----------------------------------------------------------------------------
# word2vec
# -----------------------------------------------------------------------------

def load_word2vec(path):
    
    if os.path.exists(path):
        model = word2vec.Word2Vec.load(path)
    else:
        model = None

    return model


# -----------------------------------------------------------------------------
# 文ベクトル
# -----------------------------------------------------------------------------

def sentence_to_vector(sentence, word2vec, settings):
    
    words = sentence.strip().split(" ")
    sentence_vector = np.zeros(settings["word2vec"]["dim"], dtype="float32")
    length = 0
    for word in words:
        try:
            tmp = word2vec[word]
        except:
            continue
        sentence_vector += tmp
        length += 1
    if length == 0:
        return []
    sentence_vector = sentence_vector / float(length)
    return sentence_vector


def get_train_vectors_and_labels(word2vec, settings):
    
    sentence_vectors = []
    labels = []

    f = open(settings["path"]["train"], "r")
    for line in f:
        elems = line.strip().split("\t")
        label_str = elems[0]
        sentence = elems[1]

        label = -1
        sentence_vector = []
        try:
            sentence_vector = sentence_to_vector(sentence, word2vec, settings)
        except:
            sentence_vector = []
        if label_str in settings["data"]["label_map"]:
            label = settings["data"]["label_map"][label_str]
        if label != -1 and len(sentence_vector) != 0:
            sentence_vectors.append(sentence_vector)
            labels.append(label)
    f.close()
    return [sentence_vectors, labels]

def get_test_vectors_and_sentences(word2vec, settings):
    
    sentences = []
    sentence_map = {}
    sentence_vectors = []

    f = open(settings["path"]["test"], "r")
    for line in f:
        elems = line.strip().split("\t")
        sentences.append(elems[0])
        sentence_map[elems[0]] = elems[1]
        sentence_vector = []
        try:
            sentence_vector = sentence_to_vector(line.strip(), word2vec, settings)
        except:
            sentence_vector = []
        if not len(sentence_vector) == 0:
            sentence_vectors.append(sentence_vector)
    f.close()
    return [sentence_vectors, sentences, sentence_map]


# -----------------------------------------------------------------------------
# 理由推定
# -----------------------------------------------------------------------------

def compute_distances(target_index, all_vectors, sentence_vectors, results):
    
    dist_array = []

    for i, svec in enumerate(sentence_vectors):
        
        dist = 10000.0
        nearest = -1

        for j, vec in enumerate(all_vectors):
            tmp = dis.cosine(vec, svec)
            if tmp < dist:
                dist = tmp
                nearest = j

        if target_index == nearest:
            #dist = dis.cosine(person_vector, svec)
            dist_array.append([dist, results[i]])
    
    return dist_array


def classify_eval(settings):
    
    w2v = load_word2vec(settings["path"]["word2vec"])
    train_vectors, labels = get_train_vectors_and_labels(w2v, settings)
    
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    result = cross_validate(model, train_vectors, labels, cv=skf, scoring=scoring, return_train_score=False)

    keys = result.keys()
    for key in keys:
        print(key+": "+str(result[key].mean()))


def reasoning(settings, target):
    
    w2v = load_word2vec(settings["path"]["word2vec"])
    train_vectors, labels = get_train_vectors_and_labels(w2v, settings)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=5)
    model.fit(train_vectors, labels)
    
    test_vectors, sentences, sentence_map = get_test_vectors_and_sentences(w2v, settings)

    results = model.predict_proba(test_vectors)
    new_results = []

    for i, res in enumerate(results):
        tmp = res.tolist()
        #nres.append(sentences[i])
        tmp.append(sentence_map[sentences[i]])
        new_results.append(tmp)
    
    try:
        target_index = -1
        all_vectors = []
        for i, name in enumerate(settings["data"]["names"]):
            if name == target:
                target_index = i
            all_vectors.append(w2v[name])
    except:
        return False
    
    # 犯人ベクトルに近い文を選択
    dist_res = compute_distances(target_index, all_vectors, test_vectors, new_results)
    dist_res.sort()
    
    for d in dist_res:
        print(str(d[0])+"\t"+str(d[1][0])+"\t"+str(d[1][1])+"\t"+str(d[1][2])+"\t"+str(d[1][3]))



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("-s", required=True, action="store", dest="setting_path", help="Setting file path")
    arg_parser.add_argument("-t", action="store", dest="target", help="Target climinal")
    arg_parser.add_argument("-e", action="store_true", default=False, dest="classify_eval", help="")
    args = arg_parser.parse_args()
    
    settings = ""
    with open(args.setting_path, "r") as f:
        settings = json.load(f)
    if settings == "":
        return False
    
    if args.classify_eval:
        classify_eval(settings)
    else:
        reasoning(settings, args.target)
    
    return True
    

if __name__ == "__main__":
    main()

