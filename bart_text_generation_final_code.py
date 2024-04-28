"""
NLP Project - Recipe Generator Using BART Model 
This script trains a BART model on a recipe dataset, evaluates its performance using Rouge scores,
and generates recipes based on user input ingredients.

"""

# Install required libraries
!pip install datasets
!pip install rouge_score
!pip install transformers

# Import necessary libraries
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from rouge_score import rouge_scorer

def load_recipe_data(csv_file_path):
    """
    Load recipe dataset from CSV file.

    Args:
    - csv_file_path (str): Path to the CSV file containing recipe data.

    Returns:
    - train_data (datasets.Dataset): Training dataset.
    - val_data (datasets.Dataset): Validation dataset.
    - test_data (datasets.Dataset): Test dataset.
    """
    dataset = load_dataset('csv', data_files=csv_file_path)
    total_samples = len(dataset['train'])
    train_size = int(0.95 * total_samples)
    val_size = int(0.025 * total_samples)
    test_size = total_samples - train_size - val_size

    train_data = dataset['train'].select(range(train_size))
    val_data = dataset['train'].select(range(train_size, train_size + val_size))
    test_data = dataset['train'].select(range(train_size + val_size, total_samples))

    return train_data, val_data, test_data

def train_recipe_model(train_data, val_data):
    """
    Train BART model on recipe data.

    Args:
    - train_data (datasets.Dataset): Training dataset.
    - val_data (datasets.Dataset): Validation dataset.

    Returns:
    - trainer (transformers.Trainer): Trainer object for training the model.
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['ingredients'], max_length=1024, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        labels = tokenizer(text_target=examples["directions"], max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_data.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_data.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="facebook/bart-large-cnn")

    training_args = Seq2SeqTrainingArguments(output_dir="/content/Checkpoints", evaluation_strategy="epoch", learning_rate=2e-5, predict_with_generate=True, num_train_epochs=5)

    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_val_dataset)

    trainer.train()

    return trainer

def generate_recipe_from_user_input(model, tokenizer, user_input, max_length=512, num_beams=4, temperature=1.0, top_k=50):
    """
    Generates a recipe based on user input ingredients.

    Args:
    - model (transformers.BartForConditionalGeneration): Pre-trained BART model.
    - tokenizer (transformers.BartTokenizer): Tokenizer corresponding to the BART model.
    - user_input (str): User-provided input ingredients for the recipe.
    - max_length (int): Maximum length of the generated recipe.
    - num_beams (int): Number of beams for beam search.
    - temperature (float): Temperature parameter for sampling.
    - top_k (int): Top-k parameter for sampling.

    Returns:
    - generated_recipe (str): Generated recipe based on the user input ingredients.
    """
    model_input = tokenizer(user_input, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**model_input, max_length=max_length, num_beams=num_beams, temperature=temperature, top_k=top_k, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, decoder_start_token_id=tokenizer.bos_token_id)
    generated_recipe = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_recipe

def evaluate_model_with_rouge(trainer, test_data):
    """
    Evaluate the trained model using Rouge scores.

    Args:
    - trainer (transformers.Trainer): Trainer object for the trained model.
    - test_data (datasets.Dataset): Test dataset.

    Returns:
    - rouge_scores (dict): Rouge scores for the model evaluation.
    """
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    predictions = trainer.predict(test_data)
    references = test_data['directions']
    scores = rouge.compute(predictions=predictions.predictions, references=references)
    return scores

# Main script
if __name__ == "__main__":
    # Define the path to the CSV file containing recipes
    csv_file_path = '/content/Recipes.csv'

    # Load the recipe dataset from the CSV file
    train_data, val_data, test_data = load_recipe_data(csv_file_path)

    # Train the BART model on the recipe dataset
    trained_model_trainer = train_recipe_model(train_data, val_data)

    # Evaluate the trained model using Rouge scores
    rouge_scores = evaluate_model_with_rouge(trained_model_trainer, test_data)
    print(f"Rouge Scores: {rouge_scores}")

    # Take input ingredients from the user
    user_input = input("Enter ingredients (comma-separated): ")

    # Generate a recipe based on the user input ingredients
    generated_recipe = generate_recipe_from_user_input(trained_model_trainer.model, trained_model_trainer.tokenizer, user_input)

    # Print the generated recipe
    print("Generated recipe:")
    print(generated_recipe)
