import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from torch.optim import AdamW

# Step 1: Data Preparation
# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length):
#         self.tokenizer = tokenizer
#         self.data = data
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text = self.data.iloc[idx]['Headline'] + " " + self.data.iloc[idx]['summary']
#         label = self.data.iloc[idx]['Label']
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.max_headline_length = 12
        self.max_summary_length = 60

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline = self.data.iloc[idx]['Headline']
        summary = self.data.iloc[idx]['summary']
        label = self.data.iloc[idx]['Label']

        # Encode headline
        encoding_headline = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_headline_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Encode summary
        encoding_summary = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            max_length=self.max_summary_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Combine input_ids and attention_mask for headline and summary
        input_ids = torch.cat((encoding_headline['input_ids'], encoding_summary['input_ids']), dim=1).flatten()
        attention_mask = torch.cat((encoding_headline['attention_mask'], encoding_summary['attention_mask']), dim=1).flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define function to calculate F-measure and class-wise F-measure
def compute_f_measure(labels, preds):
    f_measure_macro = f1_score(labels, preds, average='macro')
    classwise_f_measure = classification_report(labels, preds, target_names=['0', '1'])
    return f_measure_macro, classwise_f_measure

# Load CSV files
train_data = pd.read_csv('train.csv').head(10)
# train_data = train_data.head(50)
dev_data = pd.read_csv('dev.csv').head(10)
# dev_data = dev_data.head(50)
test_data = pd.read_csv('test.csv').head(10)
# test_data = test_data.head(50)

# Step 2: Model Selection
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Step 3: Training
MAX_LENGTH = 60
BATCH_SIZE = 16
EPOCHS = 1

train_dataset = CustomDataset(train_data, tokenizer)
dev_dataset = CustomDataset(dev_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
with open('validation_results_testing.txt', 'w') as f:
    # Training loop
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
        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Average Training Loss: {average_train_loss}')


        # Evaluation loop on development set
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(dev_loader)
        f_measure_macro, classwise_f_measure = compute_f_measure(val_labels, val_preds)
        f.write(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, F-measure (Macro): {f_measure_macro}\n')
        f.write('Class-wise F-measure:\n')
        f.write(classwise_f_measure)
        f.write('\n\n')
        print(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, F-measure (Macro): {f_measure_macro}')

# Step 4: Evaluation on Test Set
test_dataset = CustomDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Write predictions over the test set to a CSV file
test_results = pd.DataFrame({'Headline': test_data['Headline'], 'summary': test_data['summary'], 'Label': test_labels, 'Prediction': test_preds})
test_results.to_csv('test_predictions.csv', index=False)

print("Test predictions saved to test_predictions.csv")
