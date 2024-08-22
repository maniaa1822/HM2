#%%
import random
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
from negate import Negator
import re
import nltk
import numpy as np
nltk.download('wordnet')


class DatasetAugmenter:
    """
    Class for augmenting datasets using various techniques.
    
    Args:
        model_name (str): The name of the pre-trained model to use for contextual embeddings.
        similarity_threshold (float): The threshold value for cosine similarity between original and augmented embeddings.
        max_word_types (int): The maximum number of word types to consider for synonyms, hypernyms, and hyponyms.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", similarity_threshold=0.9, max_word_types=5):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_word_types = max_word_types
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device="cuda")
        self.model = AutoModel.from_pretrained(model_name).cuda()
        self.negator = Negator(use_gpu=True)
        self.ds = load_dataset("tommasobonomo/sem_augmented_fever_nli")
        self.df = pd.DataFrame(self.ds['train']).head(15000)

    def augment_dataset(self, dataset, augmentation_factor=10.0):
        """
        Augments the dataset using various techniques.
        
        Args:
            dataset (pandas.DataFrame): The original dataset to be augmented.
            augmentation_factor (float): The factor by which to augment the dataset.
        
        Returns:
            pandas.DataFrame: The augmented dataset.
        """
        df = self.df
        # Add synonyms, hypernyms, and hyponyms
        #self.add_syn_hyper_hypo_to_dataframe(df)
        
        # Word substitution augmentation
        #df = self.augment_dataset_word(df)
        
        # Create new neutral entries
        #num_new_neutral = int(len(df) * augmentation_factor / 2)
        num_new_neutral = 5
        new_neutral_df = self.create_new_neutral_entries(df, num_new_neutral)
        df = pd.concat([df, new_neutral_df], ignore_index=True)
        
        # Create new negative entries
        #num_new_negative = int(len(df) * augmentation_factor / 2)
        num_new_negative = 15
        new_negative_df = self.create_new_negative_entries(df, num_new_negative)
        df = pd.concat([df, new_negative_df], ignore_index=True)
        
        return df

    def add_syn_hyper_hypo_to_dataframe(self, df):
        """
        Adds synonyms, hypernyms, and hyponyms to the dataframe.
        
        Args:
            df (pandas.DataFrame): The dataframe to add the word details to.
        """
        for _, row in df.iterrows():
            if 'wsd' in row and 'hypothesis' in row['wsd']:
                for word_detail in row['wsd']['hypothesis']:
                    if 'nltkSynset' in word_detail and word_detail['nltkSynset'] != 'O':
                        synset = wn.synset(word_detail['nltkSynset'])
                        word_detail['synonyms'] = [lemma.name() for lemma in synset.lemmas() if lemma.name() != word_detail['text']][:self.max_word_types]
                        word_detail['hypernyms'] = [lemma.name().replace('_', ' ').split('.')[0] for hypernym in synset.hypernyms() for lemma in hypernym.lemmas()][:self.max_word_types]
                        word_detail['hyponyms'] = [lemma.name().replace('_', ' ').split('.')[0] for hyponym in synset.hyponyms() for lemma in hyponym.lemmas()][:self.max_word_types]
                        word_detail['antonyms'] = [lemma.name() for lemma in synset.lemmas()[0].antonyms()][:self.max_word_types]

    def get_contextual_embedding(self, text):
        """
        Computes the contextual embedding for a given text.
    
        Args:
            text (str): The input text.
    
        Returns:
            numpy.ndarray: The contextual embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def cosine_similarity(self, a, b):
        """
        Computes the cosine similarity between two vectors.
        
        Args:
            a (numpy.ndarray): The first vector.
            b (numpy.ndarray): The second vector.
        
        Returns:
            float: The cosine similarity score.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def augment_dataset_word(self, df):
        """
        Augments the dataset by substituting words with synonyms, hypernyms, and hyponyms.
        
        Args:
            df (pandas.DataFrame): The original dataset.
        
        Returns:
            pandas.DataFrame: The augmented dataset.
        """
        augmented_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            original_hypothesis = row['hypothesis']
            original_embedding = self.get_contextual_embedding(original_hypothesis)
            
            if 'wsd' in row and 'hypothesis' in row['wsd']:
                for word_detail in row['wsd']['hypothesis']:
                    if 'nltkSynset' in word_detail and word_detail['nltkSynset'] != 'O':
                        original_word = word_detail['text']
                        candidate_words = (
                            word_detail.get('synonyms', []) + 
                            word_detail.get('hypernyms', []) + 
                            word_detail.get('hyponyms', [])
                        )
                        
                        for candidate in candidate_words:
                            new_hypothesis = re.sub(r'\b{}\b'.format(re.escape(original_word)), candidate, original_hypothesis, flags=re.IGNORECASE)
                            new_embedding = self.get_contextual_embedding(new_hypothesis)
                            
                            similarity = self.cosine_similarity(original_embedding, new_embedding)
                            
                            if similarity >= self.similarity_threshold:
                                new_row = row.copy()
                                new_row['hypothesis'] = new_hypothesis
                                new_row['augmentation_info'] = {
                                    'original_word': original_word,
                                    'substituted_word': candidate,
                                    'similarity_score': float(similarity)
                                }
                                augmented_data.append(new_row)
        
        return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)

    def swap_hypothesis_and_assign_neutral(self, df, index):
        """
        Swaps the hypothesis with another random hypothesis and assigns a 'NEUTRAL' label.
        
        Args:
            df (pandas.DataFrame): The original dataset.
            index (int): The index of the entry to swap.
        
        Returns:
            pandas.Series: The modified entry.
        """
        original_entry = df.loc[index].copy()
        other_indices = df.index[df.index != index]
        random_index = random.choice(other_indices)
        
        original_entry['hypothesis'] = df.loc[random_index, 'hypothesis']
        original_entry['label'] = 'NEUTRAL'
        original_entry['wsd']['hypothesis'] = df.loc[random_index, 'wsd']['hypothesis']
        original_entry['srl']['hypothesis'] = df.loc[random_index, 'srl']['hypothesis']
        
        return original_entry

    def create_new_neutral_entries(self, df, num_new_entries):
        """
        Creates new neutral entries by swapping hypotheses with random entries.
        
        Args:
            df (pandas.DataFrame): The original dataset.
            num_new_entries (int): The number of new neutral entries to create.
        
        Returns:
            pandas.DataFrame: The new neutral entries.
        """
        new_entries = [self.swap_hypothesis_and_assign_neutral(df, random.choice(df.index)) for _ in tqdm(range(num_new_entries), desc="Creating new neutral entries")]
        return pd.DataFrame(new_entries)

    def negate_sentence(self, sentence):
        """
        Negates a given sentence.
        
        Args:
            sentence (str): The input sentence.
        
        Returns:
            str: The negated sentence.
        """
        return self.negator.negate_sentence(sentence)

    def swap_hypothesis_with_negation(self, df, index):
        """
        Swaps the hypothesis with its negation and assigns a 'CONTRADICTION' label.
        
        Args:
            df (pandas.DataFrame): The original dataset.
            index (int): The index of the entry to swap.
        
        Returns:
            pandas.Series: The modified entry.
        """
        original_entry = df.loc[index].copy()
        if original_entry['label'] == 'ENTAILMENT':
            original_entry['hypothesis'] = self.negate_sentence(original_entry['hypothesis'])
            original_entry['label'] = 'CONTRADICTION'
            #TODO compute wsd and srl for negated hypothesis and append
            #original_entry['wsd']['hypothesis'] = df.loc[index, 'wsd']['hypothesis']
            #original_entry['srl']['hypothesis'] = df.loc[index, 'srl']['hypothesis']
        return original_entry

    def create_new_negative_entries(self, df, num_new_entries):
        """
        Creates new negative entries by swapping hypotheses with negations.
        
        Args:
            df (pandas.DataFrame): The original dataset.
            num_new_entries (int): The number of new negative entries to create.
        
        Returns:
            pandas.DataFrame: The new negative entries.
        """
        new_entries = [self.swap_hypothesis_with_negation(df, random.choice(df.index)) for _ in tqdm(range(num_new_entries), desc="Creating new negative entries")]
        return pd.DataFrame(new_entries)

    def save_to_huggingface(self, dataset, dataset_name, token):
        """
        Saves the dataset to the Hugging Face Hub.
        
        Args:
            dataset (pandas.DataFrame): The dataset to be saved.
            dataset_name (str): The name of the dataset.
            token (str): The authentication token for the Hugging Face Hub.
        """
        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(dataset)
        # Create a DatasetDict
        dataset_dict = DatasetDict({"train": hf_dataset})
        # Push to Hugging Face Hub
        dataset_dict.push_to_hub(dataset_name, token=token)
# Usage example:
# augmenter = DatasetAugmenter()
# fever_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
# augmented_dataset = augmenter.augment_dataset(fever_dataset['train'])
# augmenter.save_to_huggingface(augmented_dataset, "your_dataset_name", "your_huggingface_token")
# %%
#%%
# Usage example:
if __name__ == "__main__":
    augmenter = DatasetAugmenter()
    fever_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
    augmented_dataset = augmenter.augment_dataset(fever_dataset)
    # Remove the 'srl' column using pandas
    augmented_dataset = augmented_dataset.drop(columns=['srl'])
    #augmenter.save_to_huggingface(augmented_dataset, "fever_augmented_syn_hyper_hypo", "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")


# %%
