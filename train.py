import os
import evaluate
import numpy as np
from csv import reader
from datasets import Dataset
from torch.utils.data import random_split
from transformers import logging, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

# keep terminal clear of warnings/low-level messages
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# load and preprocess tsv data from a given file path
def preprocess_data(data_path):
    import ast
    data = {"text": [], "labels": []}

    def format_label(line):
        return "[TITLE]" + line[1] + \
            "[INGREDIENTS]" + "&&".join(ast.literal_eval(line[2])) + \
            "[DIRECTIONS]" + "&&".join(ast.literal_eval(line[3]))

    with open(data_path, "r") as f:
        csv = reader(f)
        next(csv, None)  # skip csv header row
        for line in csv:
            data["text"].append(ast.literal_eval(line[6]))
            data["labels"].append(format_label(line))
    print(data["text"][1])
    print(data["labels"][1])
    # return Dataset.from_dict(data)


# tokenize the labels of a given dataset
def tokenize_labels(data):
    # tokenize the labels of a single example within the dataset
    def tokenize_label(example):
        tokenized = tokenizer(
            example["tokens"], padding=True, truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(example["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # special tokens created by is_split_into_words labeled -100 (to be ignored)
                    # https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    return data.map(tokenize_label, batched=True)


# split a given dataset into a training set and a testing set
def split_data(data, train_size=0.9):
    train_size = int(train_size * len(data))
    test_size = len(data) - train_size
    return random_split(data, [train_size, test_size])


# fine-tune a pre-trained model using tsv data from a given file path
def train(data_path, save_path="model"):
    data = preprocess_data(data_path)
    tokenized_data = tokenize_labels(data)
    train_data, test_data = split_data(tokenized_data)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    rouge = evaluate.load("rouge")

    # compute the precision, recall, f1 score, and accuracy of a given prediction
    def compute_metrics(p):
        predictions, labels = p
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        true_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        true_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        results = rouge.compute(
            predictions=true_predictions, references=true_labels, use_stemmer=True)
        """return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }"""
        return {k: round(v, 4) for k, v in results.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    preprocess_data("data/train_tiny.csv")
# train("./data/train_tiny.tsv", "model")
