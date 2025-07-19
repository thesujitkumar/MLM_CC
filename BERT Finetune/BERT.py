import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import random
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.max_headline_length = 12
        self.max_summary_length = 500

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline = str(self.data.iloc[idx]['headline'])
        summary = str(self.data.iloc[idx]['news'])
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer.encode_plus(
        text=headline,
        text_pair=summary,
        add_special_tokens=True,
        max_length=500,  # total length = headline + news
        padding='max_length',
        truncation='longest_first',  # trims longer part first
        return_attention_mask=True,
        return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_f_measure(labels, preds):
    f_measure_macro = f1_score(labels, preds, average='macro')
    classwise_f_measure = classification_report(labels, preds, target_names=['0', '1'])
    return f_measure_macro, classwise_f_measure

# Load datasets
train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# DataLoader
BATCH_SIZE = 16
EPOCHS = 200

train_dataset = CustomDataset(train_data, tokenizer)
dev_dataset = CustomDataset(dev_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
best_f1 = 0.0
best_model_path = "best_bert_model.pt"

with open('validation_results.txt', 'w') as f:
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f'Validating {epoch+1}/{EPOCHS}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(dev_loader)
        f_macro, classwise_f = compute_f_measure(val_labels, val_preds)

        f.write(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Macro F1: {f_macro:.4f}\n')
        f.write('Class-wise F1:\n' + classwise_f + '\n\n')

        print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Macro F1: {f_macro:.4f}')

        if f_macro > best_f1:
            best_f1 = f_macro
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with F1: {f_macro:.4f}")

# Load best model for testing
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Test
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        preds = torch.argmax(outputs.logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Save predictions
test_results = pd.DataFrame({
    'headline': test_data['headline'],
    'news': test_data['news'],
    'label': test_labels,
    'Prediction': test_preds
})
test_results.to_csv('test_predictions.csv', index=False)
print("Test predictions saved using best dev model.")
