import pandas as pd
import torch
import csv

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Add special tokens to the tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['similar', 'unlike']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline = str(self.data.iloc[idx]['Headline'])
        body = str(self.data.iloc[idx]['Body'])
        label = int(self.data.iloc[idx]['Label'])

        if len(body.split()) > 200:
            body = body[:200]

        # Construct input text based on the label
        if label == 0:
            combined_text = f"{headline} {body} This given headline and body pair is similar"
            target_text = "similar"
        else:
            combined_text = f"{headline} {body} This given headline and body pair is unlike"
            target_text = "unlike"

        # Tokenize input text
        inputs = self.tokenizer(combined_text, return_tensors="pt")

        # Re-tokenize input text with max_length
        inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = inputs.input_ids.flatten()

        # Pad input_ids with zeros on the left
        pad_length = self.max_length - len(input_ids)
        padded_input_ids = torch.cat([torch.zeros(pad_length, dtype=torch.long), input_ids])

        # Find the position of the last occurrence of "similar" or "unlike" in the tokens
        tokens = self.tokenizer.convert_ids_to_tokens(padded_input_ids)
        last_similar_index = max((i for i, token in enumerate(tokens) if token == 'similar'), default=-1)
        last_unlike_index = max((i for i, token in enumerate(tokens) if token == 'unlike'), default=-1)

        # Mask the last occurrence of "similar" or "unlike"
        if last_similar_index > last_unlike_index:
            masked_position = last_similar_index
        else:
            masked_position = last_unlike_index

        labels = torch.zeros_like(padded_input_ids)
        labels[masked_position] = self.tokenizer.mask_token_id

        return {
            'input_ids': padded_input_ids,
            'masked_position': masked_position,
            'labels': labels,
            'target': target_text
        }

# Load train, dev, and test datasets
train_data = pd.read_csv("train.csv")
# train_data = train_data.head(200)
dev_data = pd.read_csv("dev.csv")
# dev_data = dev_data.head(50)
test_data = pd.read_csv("test.csv")
# test_data = test_data.head(50)

# Define your tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

# Define parameters
batch_size = 16
max_length = 500
learning_rate = 5e-5
epochs = 200

# Prepare your datasets
train_dataset = CustomDataset(train_data, tokenizer, max_length)
dev_dataset = CustomDataset(dev_data, tokenizer, max_length)
test_dataset = CustomDataset(test_data, tokenizer, max_length)

# Define your data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        masked_position = batch['masked_position'].to(device)
        labels = batch['labels'].to(device)
        # print(labels)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

    # Evaluation loop (on dev set)
    # Evaluation loop (on dev set)
    model.eval()
    similar_probabilities = []
    unlike_probabilities = []
    for batch in tqdm(dev_loader, desc="Dev Evaluation"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            masked_position = batch['masked_position'].to(device)
            labels = batch['labels']
            target_text = batch['target']

            # Generate logits for each input
            logits = model(input_ids=input_ids).logits

            # Ensure masked position is within bounds
            masked_position = masked_position.clamp(max=logits.size(1)-1)

            # Calculate the probability of the generated text being similar or unlike
            similar_probability = torch.exp(logits[range(logits.shape[0]), masked_position][:, tokenizer.convert_tokens_to_ids('similar')])
            unlike_probability = torch.exp(logits[range(logits.shape[0]), masked_position][:, tokenizer.convert_tokens_to_ids('unlike')])
            similar_probabilities.extend([(tokenizer.decode(input_ids[i, masked_position].cpu().numpy()), target_text[i], prob.item()) for i, prob in enumerate(similar_probability)])
            unlike_probabilities.extend([(tokenizer.decode(input_ids[i, masked_position].cpu().numpy()), target_text[i], prob.item()) for i, prob in enumerate(unlike_probability)])






    # Write probabilities to CSV files
    output_file_similar = "test_dev_similar_probabilities.csv"
    with open(output_file_similar, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generated Text", "Target Label", "Similar Probability"])
        for text, target, probability in similar_probabilities:
            writer.writerow([text, target, probability])

    output_file_unlike = "test_dev_unlike_probabilities.csv"
    with open(output_file_unlike, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generated Text", "Target Label", "Unlike Probability"])
        for text, target, probability in unlike_probabilities:
            writer.writerow([text, target, probability])

    print(f"Similar probabilities saved to {output_file_similar}.")
    print(f"Unlike probabilities saved to {output_file_unlike}.")



    # Evaluation loop (on dev set)
    # Evaluation loop (on dev set)
    # model.eval()
    similar_probabilities = []
    unlike_probabilities = []
    for batch in tqdm(test_loader, desc="test Evaluation"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            masked_position = batch['masked_position'].to(device)
            labels = batch['labels']
            target_text = batch['target']

            # Generate logits for each input
            logits = model(input_ids=input_ids).logits

            # Ensure masked position is within bounds
            masked_position = masked_position.clamp(max=logits.size(1)-1)

            # Calculate the probability of the generated text being similar or unlike
            similar_probability = torch.exp(logits[range(logits.shape[0]), masked_position][:, tokenizer.convert_tokens_to_ids('similar')])
            unlike_probability = torch.exp(logits[range(logits.shape[0]), masked_position][:, tokenizer.convert_tokens_to_ids('unlike')])
            similar_probabilities.extend([(tokenizer.decode(input_ids[i, masked_position].cpu().numpy()), target_text[i], prob.item()) for i, prob in enumerate(similar_probability)])
            unlike_probabilities.extend([(tokenizer.decode(input_ids[i, masked_position].cpu().numpy()), target_text[i], prob.item()) for i, prob in enumerate(unlike_probability)])






    # Write probabilities to CSV files
    output_file_similar = "test_test_similar_probabilities.csv"
    with open(output_file_similar, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generated Text", "Target Label", "Similar Probability"])
        for text, target, probability in similar_probabilities:
            writer.writerow([text, target, probability])

    output_file_unlike = "test_test_unlike_probabilities.csv"
    with open(output_file_unlike, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generated Text", "Target Label", "Unlike Probability"])
        for text, target, probability in unlike_probabilities:
            writer.writerow([text, target, probability])

    print(f"Similar probabilities saved to {output_file_similar}.")
    print(f"Unlike probabilities saved to {output_file_unlike}.")
