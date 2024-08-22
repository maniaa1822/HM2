# %%
import torch
import numpy as np
import pandas as pd
from typing import Dict
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


### Model Parameters
# we will use with Distil-BERT
language_model_name = "roberta-base"

### Training Argurments

# this GPU should be enough for this task to handle 32 samples per batch
batch_size = 16

# optim
learning_rate = 1e-4
weight_decay = 0.001  # we could use e.g. 0.01 in case of very low and very high amount of data for regularization

# training
epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


set_seed(42)

# %% load the dataset
label_mapping = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}
fever_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
fever_augmented = load_dataset("matteo1822/fever_augmented")
#shuffle the dataset
#%%
fever_dataset = fever_dataset.shuffle(seed=42)
fever_augmented = fever_augmented.shuffle(seed=42)
sample_fever = pd.DataFrame(fever_augmented['train'][:300])
eval_fever = pd.DataFrame(fever_dataset["validation"][:100])
#%%
fever_dataset
sample_fever
# %%
from datasets import load_metric

# Metrics
# %%
from sklearn.metrics import accuracy_score, f1_score

from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score from EvalPrediction object.

    Parameters:
    - eval_pred: EvalPrediction, an object containing the predictions and labels.

    Returns:
    - A dictionary containing 'accuracy' and 'f1' scores.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    # If your model outputs logits, you need to convert these logits to class indices
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(
        labels, predictions, average="weighted"
    )  # Use 'weighted' to handle imbalanced classes
    return {"accuracy": accuracy, "f1": f1}


# %%
## Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    language_model_name, num_labels=3
)  # number of the classes

tokenizer = AutoTokenizer.from_pretrained(language_model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokenized["label"] = [label_mapping[label] for label in examples["label"]]
    return tokenized


fever_dataset_sample_from_pandas = Dataset.from_pandas(sample_fever)
fever_dataset_eval_from_pandas = Dataset.from_pandas(eval_fever)

tokenized_train_dataset = fever_dataset_sample_from_pandas.map(
    tokenize_function, batched=True
)
tokenized_eval_dataset = fever_dataset_eval_from_pandas.map(
    tokenize_function, batched=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
tokenized_train_dataset
# tokenized_train_dataset['input_ids'][0]
# tokenized_train_dataset['label'][0]
# tokenized_train_dataset['label'][0]
# %% convert back to pandas
# tokenized_train_dataset.to_pandas()
# tokenized_train_dataset['input_ids'][0]
# decode
# tokenizer.decode(tokenized_train_dataset['input_ids'][0])
# tokenized_train_dataset
# %%
from transformers import TrainingArguments, Trainer

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for model checkpoints
    num_train_epochs=epochs,  # Number of training epochs, you've set it to 1
    per_device_train_batch_size=batch_size,  # Batch size for training
    per_device_eval_batch_size=batch_size,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,  # Weight decay if we apply some.
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # The instantiated 🤗 Transformers model to be trained
    args=training_args,  # Training arguments, defined above
    train_dataset=tokenized_train_dataset,  # Training dataset
    eval_dataset=tokenized_eval_dataset,  # Evaluation dataset, you can define one
    tokenizer=tokenizer,  # The tokenizer used for encoding the data
    data_collator=data_collator,  # The function used for collating batches
    compute_metrics=compute_metrics,  # The function that will be used to compute metrics
)

# %% Train the model

trainer.train()
# %%
trainer.evaluate()
# %% 