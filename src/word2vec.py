#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json

from gensim.models import word2vec


def word2vec_model_name(settings):
    sg = "cbow"
    if settings["word2vec"]["skip_gram"]:
        sg = "skip-gram"
    dim = str(settings["word2vec"]["dim"])
    window = str(settings["word2vec"]["window"])
    min_count = str(settings["word2vec"]["min_count"])
    ite = str(settings["word2vec"]["iter"])
    name = "w2v_"+sg+"_"+dim+"_"+window+"_"+min_count+"_"+ite+".model"

    return name

def train_word2vec(wordlists, settings):
    
    sg = 0
    if settings["word2vec"]["skip_gram"]:
        sg = 1
    model = word2vec.Word2Vec(wordlists,
                              sg=sg,
                              size=settings["word2vec"]["dim"],
                              min_count=settings["word2vec"]["min_count"],
                              window=settings["word2vec"]["window"],
                              iter=settings["word2vec"]["iter"])
    model.save("model/"+word2vec_model_name(settings))
    return model

def main():
    
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("-s", required=True, action="store", dest="setting_path", help="Setting file path")
    args = arg_parser.parse_args()

    settings = ""
    with open(args.setting_path, "r") as f:
        settings = json.load(f)
    
    wordlists = []
    for line in sys.stdin:
        wordlist = line.strip().split(" ")
        wordlists.append(wordlist)

    train_word2vec(wordlists, settings)
    


if __name__ == "__main__":
    main()
