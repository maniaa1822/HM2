import argparse
import os
import shutil
import sys
from typing import Dict
import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset

from transformers import (AdamW, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EvalPrediction, Trainer,
                          TrainingArguments)

# Mapping of labels to integers
label_to_int = {
    "ENTAILMENT": 0,
    "CONTRADICTION": 1,
    "NEUTRAL": 2,
}

# Load evaluation metrics
load_accuracy = evaluate.load("accuracy")
load_precision = evaluate.load("precision")
load_f1 = evaluate.load("f1")
load_recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Args:
        eval_pred (tuple): Tuple containing logits and labels.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    precision = load_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    recall = load_recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    return {"accuracy": accuracy, "precision": precision, "f1": f1, "recall": recall}


def get_model(model_name):
    """
    Retrieves a pre-trained model for sequence classification.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        model (AutoModelForSequenceClassification): The loaded pre-trained model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        data_collator (DataCollatorWithPadding): The data collator for padding sequences.
        tokenize_function (function): The function used for tokenizing input examples.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        output_attentions=False,
        output_hidden_states=False,
        num_labels=3,
    )  # number of classes

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Freeze certain layers of the model
    for name, param in model.named_parameters():
        is_embedding = name.startswith("distilbert.embeddings")
        freeze_threshold = 3
        mha_freeze = (
            name.startswith("distilbert.transformer")
            and int(name.split(".")[3]) > freeze_threshold
        )
        is_last = name.startswith("classifier") or name.startswith("pre_classifier")
        if not (is_last):
            param.requires_grad = False
        else:
            param.requires_grad = True

    def tokenize_function(examples):
        """
        Tokenize input examples.

        Args:
            examples (dict): Dictionary containing input examples.

        Returns:
            dict: Tokenized input examples.
        """
        out = tokenizer(
            [
                f"{premise} {tokenizer.sep_token} {hypothesis}"
                for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])
            ],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        out["label"] = torch.tensor([label_to_int[label] for label in examples["label"]])
        return out

    return model, tokenizer, data_collator, tokenize_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="1822363-main.py",
        description="A program to train distilbert on fiver",
        epilog="",
    )

    parser.add_argument("action", choices=["train", "test"])
    parser.add_argument("--data", choices=["original", "adversarial"])

    res = parser.parse_args(sys.argv[1:])

    # enable tf32 support (for NVIDIA Ampere GPUs)
    use_tf32 = torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    language_model_name = "distilbert-base-uncased"
    model_path = f"model_trained_{res.data}"
    learning_rate = 1e-5
    weight_decay = 0.001
    epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if res.action == "test":
        if not os.path.exists(model_path):
            raise Exception("You are testing the model without having trained it first")
        language_model_name = model_path

    fiver = load_dataset("tommasobonomo/sem_augmented_fever_nli")
    model, tokenizer, collator, tokenize_function = get_model(language_model_name)

    if res.data == "original":
        tokenized_fiver = fiver.map(
            tokenize_function,
            batch_size=512,
            batched=True,
        )
        fiver_train = tokenized_fiver["train"].select(range(15000)).shuffle(seed=42)
        fiver_valid = tokenized_fiver["validation"]
        fiver_test = tokenized_fiver["test"]
    elif res.data == "adversarial":
        fiver_adv_train = load_dataset("matteo1822/fever_augmented_syn_hyper_hypo")['train'].shuffle(seed=42)
        #print training on "matteo1822/fever_augmented_syn_hyper_hypo"
        print('training on "matteo1822/fever_augmented_syn_hyper_hypo')
        fiver_train = fiver_adv_train.map(
            tokenize_function,
            batch_size=512,
            batched=True,
        )
        fiver_valid = fiver["validation"].map(
            tokenize_function,
            batch_size=512,
            batched=True,
        )
        fiver_adv_test = load_dataset("iperbole/adversarial_fever_nli")
        fiver_test = fiver_adv_test["test"].map(
            tokenize_function,
            batch_size=512,
            batched=True,
        )

    training_args = TrainingArguments(
        output_dir=f"{language_model_name}_checkpoints_{res.data}",
        num_train_epochs=epochs,
        auto_find_batch_size=True,
        weight_decay=weight_decay,
        save_strategy="epoch",
        learning_rate=learning_rate,
        tf32=use_tf32,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fiver_train,
        eval_dataset=fiver_valid,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if res.action == "train":
        trainer.train()
        print(trainer.evaluate(fiver_valid))
        trainer.save_model(model_path)
        #trainer.push_to_hub()

    elif res.action == "test":
        #trainer.push_to_hub
        trainer.save_model
        metrics = trainer.predict(fiver_test).metrics
        print(metrics)
        with open(f"results_test_{res.data}.tsv", "w") as f:
            l1 = "\t".join(metrics.keys())
            l2 = "\t".join([str(metrics[k]) for k in metrics.keys()])
            f.write(l1 + "\n" + l2)

#terminal command
# python 1822363-main.py train --data original
# python 1822363-main.py test --data original
# python 1822363-main.py train --data adversarial
# python 1822363-main.py test --data adversarial
