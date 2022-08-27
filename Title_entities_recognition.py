import spacy
from spacy.training import Example
import json
import random

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def train_spacy(data, iterations):
    ner = ''
    TRAIN_DATA = data
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != ner]
    with nlp.disable_pipes(*other_pipes):
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update(
                    [example],
                    drop=0.2,
                    losses=losses
                )
            print(losses)
    return nlp


TRAIN_DATA = load_data("training_data.json")
nlp = train_spacy(TRAIN_DATA, 30)
nlp.to_disk("hp_ner_model")
