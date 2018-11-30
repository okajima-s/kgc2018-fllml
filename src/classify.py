#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import sys
import json
import argparse

import numpy as np

from gensim.models import word2vec
import scipy.spatial.distance as dis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


# -----------------------------------------------------------------------------
# word2vec
# -----------------------------------------------------------------------------

def load_word2vec(path):
    
    if os.path.exists(path):
        model = word2vec.Word2Vec.load(path)
    else:
        model = None
    
    return model


def get_words(model, persons, criminals, victims, names):
    
    words = []
    for person in persons:
        try:
            wvec = model[person]
        except:
            continue
        if person in names:
            # まだらの紐の登場人物は学習に含めない
            continue
        if person in criminals:
            words.append([person, wvec, "r", 0])
        elif person in victims:
            words.append([person, wvec, "c", 1])
        else:
            words.append([person, wvec, "y", 2])
    return words


# -----------------------------------------------------------------------------
# 分類
# -----------------------------------------------------------------------------

def classify_eval(words, w2v):
    
    arr = np.zeros((len(words),words[0][1].shape[0]), dtype="float32")
    y = np.zeros(len(words), dtype="int32")
    for i, word in enumerate(words):
        arr[i] += word[1]
        y[i] = word[3]

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    result = cross_validate(model, arr, y, cv=skf, scoring=scoring, return_train_score=False)

    keys = result.keys()
    #keys.sort()
    for key in keys:
        print(key+": "+str(result[key].mean()))


def classify(words, w2v, names):
    
    arr = np.zeros((len(words),words[0][1].shape[0]), dtype="float32")
    y = np.zeros(len(words), dtype="int32")
    for i, word in enumerate(words):
        arr[i] += word[1]
        y[i] = word[3]

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=5)
    model.fit(arr, y)

    test = np.zeros((len(names),words[0][1].shape[0]), dtype="float32")
    for i, name in enumerate(names):
        test[i] += w2v[name]

    result = model.predict_proba(test)
    
    for i, res in enumerate(result):
        print(names[i]+": "+str(res))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("-s", required=True, action="store", dest="setting_path", help="Setting file path")
    arg_parser.add_argument("-e", action="store_true", default=False, dest="classify_eval", help="")
    args = arg_parser.parse_args()
    
    settings = ""
    with open(args.setting_path, "r") as f:
        settings = json.load(f)
    
    if settings == "":
        return False
    
    w2v_model = load_word2vec(settings["path"]["word2vec"])
    words = get_words(w2v_model, settings["data"]["persons"], settings["data"]["criminals"], settings["data"]["victims"], settings["data"]["names"])
    
    if args.classify_eval:
        classify_eval(words, w2v_model)
    else:
        classify(words, w2v_model, settings["data"]["names"])
    
    return True
    

if __name__ == "__main__":
    main()

