import copy
import random as rand
import sys

import nltk
import tqdm

nltk.download("wordnet")
import math
import os
from multiprocessing import Pool

from datasets import Dataset, concatenate_datasets, load_dataset
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, pipeline
import nlpaug.augmenter.word as naw

ENTAILMENT = "ENTAILMENT"
CONTRADICTION = "CONTRADICTION"
NEUTRAL = "NEUTRAL"


class SRLAugmentation:
    def __init__(self):
        self.__FRAME_BLACKLIST = [
            "AUXILIARY",
        ]

        self.__POS_BLACKLIST = ["PUNCT", "AUX", "NUM", "ADP", "DET", "PRON"]
        self.synonym_augmentation = naw.SynonymAug()

    def __get_matching_frames(self, sample):
        """Given the sample, look for matching frame names in hyp and premise"""
        matching_frames = []
        premise_ann = sample["srl"]["premise"]["annotations"]
        hypothesis_ann = sample["srl"]["hypothesis"]["annotations"]
        for ann in hypothesis_ann:
            frame_name = ann["verbatlas"]["frameName"]
            if frame_name in self.__FRAME_BLACKLIST:
                continue
            for pre_ann in premise_ann:
                # if pre_ann["verbatlas"]["frameName"] != frame_name:
                # continue
                ann_roles = ann["verbatlas"]["roles"]
                pre_ann_roles = pre_ann["verbatlas"]["roles"]
                roles_hyp = sorted([r["role"] for r in ann["verbatlas"]["roles"]])
                roles_prem = sorted([r["role"] for r in pre_ann["verbatlas"]["roles"]])
                if roles_hyp == roles_prem:
                    matching_frames.append(pre_ann)
        return matching_frames

    def __change_attribute(self, sample, frame):
        copy_sample = sample.copy()
        roles = frame["verbatlas"]["roles"]
        attr = None
        for role in roles:
            if role["role"] == "Attribute":
                attr = role
                break

        span = attr["span"]
        tokens = [
            s["rawText"]
            for s in copy_sample["srl"]["premise"]["tokens"][span[0] : span[1]]
        ]
        new_words = []
        max_substitutions = 3
        for wsd in copy_sample["wsd"]["premise"]:
            if wsd["text"] in tokens and (not wsd["pos"] in self.__POS_BLACKLIST):
                synonyms = wn.synsets(wsd["text"])
                if len(synonyms) == 0:
                    continue
                new_word = None
                for i in range(len(synonyms)):
                    lemmas = synonyms[i].lemma_names()
                    for lemma in lemmas:
                        if lemma != wsd["text"]:
                            new_word = lemma
                if new_word == None:
                    new_word = wsd["text"]
                new_phrase = new_word.split("_")
                new_words.append(new_phrase)
            else:
                new_words.append(wsd["text"])

        words = []
        for w in new_words:
            if type(w) == list:
                words += w
            else:
                words.append(w)
        copy_sample["premise"] = " ".join(words)
        return copy_sample

    def __agent_patient_swap(self, sample, frame):
        roles = frame['verbatlas']['roles']
        agent, patient = None, None
        for role in roles:
            if role['role'] == 'Agent':
                agent = role
            if role['role'] == 'Patient':
                patient = role
        agent_token_span = agent["span"]
        patient_token_span = patient["span"]
        # print(frame)
        sentence_tokens = sample['srl']['premise']['tokens']

        agent_tokens = sentence_tokens[agent_token_span[0] : agent_token_span[1]]
        patient_tokens = sentence_tokens[patient_token_span[0] : patient_token_span[1]]

        sentence_tokens[agent_token_span[0] : agent_token_span[1]] = patient_tokens
        sentence_tokens[patient_token_span[0] : patient_token_span[1]] = agent_tokens

        sentence_tokens = list(map(lambda token: token["rawText"], sentence_tokens))

        new_sentence = " ".join(sentence_tokens)
        new_sample = copy.deepcopy(sample)
        new_sample['premise'] = new_sentence
        label = sample['label']
        self.__swap_label(new_sample)
        return new_sample


    def __swap_label(self, sample):
        if sample['label'] == CONTRADICTION:
            sample["label"] = ENTAILMENT
        elif sample['label'] == ENTAILMENT: 
            sample["label"] = CONTRADICTION



    def __auxiliary_negation(self, sample):
        auxiliary_instances = []
        hyp_srl = sample['srl']['hypothesis']
        for ann in hyp_srl['annotations']:
            if ann['verbatlas']['frameName'] == 'AUXILIARY':
                new_tokens = list(map(lambda x: x.copy(), hyp_srl['tokens']))
                for tok in new_tokens:
                    text = tok['rawText']
                    if text in ('is', 'has', 'have', 'are', 'am'):
                        tok['rawText'] += 'not'
                new_sample = copy.deepcopy(sample)
                new_sample['hypothesis'] = " ".join(map(lambda x: x['rawText'], new_tokens))
                self.__swap_label(new_sample)
                yield new_sample
                

    def __build_synonyms(sample, frame, key='hypothesis'):
        frame_name = frame['frameName']
        pass


    def __call__(self, sample):
        new_sample = sample.copy()
        matching_frames = self.__get_matching_frames(sample)
        if len(matching_frames) == 0:
            return []
        # Get the tokens that match the srl
        tokens = []
        new_samples = []
        for frame in matching_frames:
            role_names = [role["role"] for role in frame["verbatlas"]["roles"]]
            # spans = [idx["span"] for idx in frame["verbatlas"]["roles"]]
            if "Attribute" in role_names:
                new_samples.append(self.__change_attribute(new_sample, frame))
            if "Agent" in role_names and "Patient" in role_names:
                new_samples.append(self.__agent_patient_swap(new_sample, frame))
        new_samples += self.__auxiliary_negation(sample)
        return new_samples


if __name__ == "__main__":
    fiver = load_dataset("./data/fiver")

    new_ds = {}
    ds = fiver["train"]

    srl_augmentation = SRLAugmentation()

    for sample in tqdm.tqdm(ds):
        augmented_samples = srl_augmentation(sample)

        print(len(augmented_samples))
        for sample in augmented_samples:
            # print(sample)
            print(type(sample))
            print(sample['hypothesis'])
            print(synonym_augmentation.augment(sample['hypothesis']))
            augmented_samples.append({
                'premise': synonym_augmentation.augment(sample["premise"])[0],
                'hypothesis': synonym_augmentation.augment(sample["hypothesis"])[0],
                'label': sample["label"]
            })
            break

        break
        for augmented_sample in augmented_samples:
            for k in ["premise", "hypothesis", "label"]:
                if not new_ds.get(k):
                    new_ds[k] = []
                new_ds[k].append(augmented_sample[k])


    new_dataset = Dataset.from_dict(new_ds)
    ds = concatenate_datasets([ds, new_dataset])
    print(ds)
    print(len(ds))

    ds.save_to_disk("./fiver-augmented")
