import torch
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling,Trainer, TrainingArguments
from datasets import load_dataset, Dataset, load_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from random import sample
from tqdm import tqdm
from torch.optim import AdamW
import evaluate
import numpy as np
import re
import logging


recipie_data = load_dataset('csv', data_files="/content/Recipes.csv")
total_length = len(recipie_data['train'])
train_size = int(0.95 * total_length)
val_size = int(0.025 * total_length)
test_size = total_length - train_size - val_size
train_set = Dataset.from_dict(recipie_data['train'][:train_size])
val_set = Dataset.from_dict(recipie_data['train'][train_size:train_size + val_size])
test_set = Dataset.from_dict(recipie_data['train'][train_size + val_size:])

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

predictions = []
for example in test_set:
    prompt = example["ingredients"]
    output = generator(prompt, max_length=140, num_return_sequences=1)[0]["generated_text"]
    predictions.append(output)

print(test_set['ingredients'][0])
print(predictions[0])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

rouge_scores = []
for i, pred in enumerate(predictions):
    reference = test_set["directions"][i]
    scores = scorer.score(reference, pred)
    rouge_scores.append(scores)

print(rouge_scores)

avg_rouge1_f1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rougeL_f1 = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

print("Average ROUGE-1 F1 Score:", avg_rouge1_f1)
print("Average ROUGE-L F1 Score:", avg_rougeL_f1)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("gpt2")


def tokenize_function(examples):
    # Tokenize inputs and outputs, and ensure truncation and padding
    inputs = examples["ingredients"]
    outputs = examples["directions"]

    # Ensure inputs and outputs are truncated and padded
    inputs_outputs = tokenizer(
        inputs,
        outputs,
        padding=True,
        truncation=True,
        max_length=20,  # Adjust as needed
        return_tensors="pt"
    )

    return inputs_outputs

# def tokenize_function(examples):
#     inputs = tokenizer(examples["ingredients"], padding="max_length", truncation=True, add_special_tokens=True, max_length=1024)
#     targets = tokenizer(examples["directions"], padding="max_length", add_special_tokens=True, truncation=True, max_length=128)

#     return {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "labels": targets["input_ids"]
#     }

tokenized_train_dataset = train_set.map(tokenize_function, batched=True)
tokenized_val_dataset = val_set.map(tokenize_function, batched=True)
tokenized_test_dataset = test_set.map(tokenize_function, batched=True)


small_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
small_val_dataset = tokenized_val_dataset.rename_column("label", "labels")
small_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

# small_train_dataset = tokenized_train_dataset
# small_val_dataset = tokenized_val_dataset
# small_test_dataset = tokenized_test_dataset


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="/content/Checkpoint",
    learning_rate = 2e-3,
    evaluation_strategy="epoch",
    eval_steps=500,
    logging_steps=500,
    save_steps=500,
    save_strategy = "epoch",
    num_train_epochs=8,
    load_best_model_at_end = True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_val_dataset,
    data_collator = data_collator
)

trainer.train()


predictions = trainer.predict(small_test_dataset)
# predictions = predictions.predictions.reshape(2,50,200)

trainer.evaluate(eval_dataset=small_test_dataset)

predicted_sentences = []

for prediction in predictions.predictions[0]:
    # Cast the float prediction values to integers before decoding
    prediction = np.int64(prediction * 0.01)
    decoded_sentence = tokenizer.decode(prediction, skip_special_tokens=True)
    # Append the decoded sentence to the list of predicted sentences
    predicted_sentences.append(decoded_sentence)



rouge = evaluate.load('rouge')
references = test_set['directions']
scores = rouge.compute(predictions=predicted_sentences, references=references[:25], use_stemmer=True)
print(f"Rouge Scores: {scores}")