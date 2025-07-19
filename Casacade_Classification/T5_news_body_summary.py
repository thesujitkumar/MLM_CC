import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def batch_summarize_news(news_bodies, model, tokenizer, device, batch_size=8):
    summaries = []
    total_batches = len(news_bodies) // batch_size + (len(news_bodies) % batch_size > 0)
    for i in tqdm(range(0, len(news_bodies), batch_size), total=total_batches, desc="Batch Processing"):
        batch_news = news_bodies[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(["summarize: " + news_body for news_body in batch_news], return_tensors="pt", max_length=512, truncation=True, pad_to_max_length=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=12, length_penalty=2.0, num_beams=4, early_stopping=True)
        batch_summaries = [tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]
        summaries.extend(batch_summaries)
    return summaries

def read_csv(filename):
    return pd.read_csv(filename)

def summarize_csv_file(input_csv_file, output_csv_file, model, tokenizer, device, batch_size=32):
    # Read the CSV file
    df = read_csv(input_csv_file)
    news_bodies = df['Body'].tolist()

    # Batch processing of news bodies
    summaries = batch_summarize_news(news_bodies, model, tokenizer, device, batch_size=batch_size)

    # Add summaries to the DataFrame
    df['summary'] = summaries

    # Save the DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to('cuda:1')

# Example usage


train_csv_file = 'FNC_Bin_Mix_Train.csv'
test_csv_file = 'FNC_Bin_Mix_Test.csv'
dev_csv_file = 'FNC_Bin_Mix_Dev.csv'
output_train_csv_file = 'TEST_FNC_T5_train_summaries.csv'
output_test_csv_file = 'Test_FNC_T5_test_summaries.csv'
output_dev_csv_file = 'Test_FNC_T5_dev_summaries.csv'

# Summarize the train, test, and dev CSV files
summarize_csv_file(dev_csv_file, output_dev_csv_file, model, tokenizer, 'cuda:1')
summarize_csv_file(train_csv_file, output_train_csv_file, model, tokenizer, 'cuda:1')
summarize_csv_file(test_csv_file, output_test_csv_file, model, tokenizer, 'cuda:1')
# summarize_csv_file(dev_csv_file, output_dev_csv_file, model, tokenizer, 'cuda')
