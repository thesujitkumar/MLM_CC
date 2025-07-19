import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Define your dataset class (implement the __len__ and __getitem__ methods)
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=250):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]["Body"]
        encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# Load pre-trained model and tokenizer
model_name = "describeai/gemini"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda:1')

# Load datasets
train_data = pd.read_csv('FNC_Bin_Mix_Train.csv').head(10)
val_data = pd.read_csv('FNC_Bin_Mix_Dev.csv').head(10)
test_data = pd.read_csv('FNC_Bin_Mix_Test.csv').head(10)

train_dataset = SummarizationDataset(train_data, tokenizer)
val_dataset = SummarizationDataset(val_data, tokenizer)
test_dataset = SummarizationDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Generate summaries for train, dev, and test CSV files
def generate_summaries(loader, output_csv_file):
    model.eval()
    summaries = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Summaries"):
            input_ids = batch["input_ids"].to('cuda:1')
            attention_mask = batch["attention_mask"].to('cuda:1')

            # Generate summaries using the model
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=60, num_beams=4, length_penalty=2.0, early_stopping=True)
            generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            summaries.extend(generated_summaries)

    # Add summaries to the DataFrame
    data = pd.DataFrame({"summary": summaries})
    data.to_csv(output_csv_file, index=False)

# Output file names
output_train_csv_file = 'test_gemini_FNC_fake_train_summaries.csv'
output_val_csv_file = 'test_gemini_FNC_fake__val_summaries.csv'
output_test_csv_file = 'test_gemini_FNC_fake_test_summaries.csv'

# Generate summaries for train, dev, and test CSV files
generate_summaries(test_loader, output_test_csv_file)
generate_summaries(train_loader, output_train_csv_file)
generate_summaries(val_loader, output_val_csv_file)
# generate_summaries(test_loader, output_test_csv_file)
