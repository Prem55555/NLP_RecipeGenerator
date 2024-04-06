# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from datasets import load_dataset
# from torch.utils.data import Dataset, DataLoader

# # Load pre-trained GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # Set padding token
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Load dataset
# dataset = load_dataset("csv", data_files="full_dataset.csv")

# # Define dataset class
# class RecipesDataset(Dataset):
#     def __init__(self, dataset, tokenizer, max_length=1024):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.recipes = dataset["train"]

#     def __len__(self):
#         return len(self.recipes)

#     def __getitem__(self, idx):
#         recipe = self.recipes[idx]
#         ingredients = recipe["ingredients"]
#         directions = recipe["directions"]

#         # Encode ingredients and directions
#         input_text = f"Title: {recipe['title']}\nIngredients: {ingredients}\nDirections: {directions}"
#         input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

#         return input_ids

# # Create dataset and dataloader
# recipes_dataset = RecipesDataset(dataset, tokenizer)
# dataloader = DataLoader(recipes_dataset, batch_size=4, shuffle=True)

# # Fine-tune the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.train()

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# num_epochs = 1
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         input_ids = batch.to(device)
#         labels = input_ids.clone()
#         labels[input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens for calculating loss
#         outputs = model(input_ids=input_ids, labels=labels)
#         loss = outputs.loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{num_epochs} completed.")

# # Save fine-tuned model
# model.save_pretrained("fine_tuned_gpt2_recipe_model")
# tokenizer.save_pretrained("fine_tuned_gpt2_recipe_model")



import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import time

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset
dataset = load_dataset("csv", data_files="full_dataset.csv")

# Define dataset class
class RecipesDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.recipes = dataset["train"]

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        ingredients = recipe["ingredients"]
        directions = recipe["directions"]

        # Encode ingredients and directions
        input_text = f"Title: {recipe['title']}\nIngredients: {ingredients}\nDirections: {directions}"
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids

# Create dataset and dataloader
recipes_dataset = RecipesDataset(dataset, tokenizer)
dataloader = DataLoader(recipes_dataset, batch_size=4, shuffle=True)

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
for epoch in range(num_epochs):
    start_time = time.time()  # Start timer for epoch
    rows_processed = 0  # Initialize the counter for rows processed
    for batch in dataloader:
        input_ids = batch.to(device)
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens for calculating loss
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rows_processed += input_ids.size(0)  # Increment the counter by batch size

    epoch_time = time.time() - start_time  # Calculate total time taken for epoch
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
    print(f"Rows processed per second: {rows_processed / epoch_time:.2f}")

# Save fine-tuned model
model.save_pretrained("fine_tuned_gpt2_recipe_model")
tokenizer.save_pretrained("fine_tuned_gpt2_recipe_model")
