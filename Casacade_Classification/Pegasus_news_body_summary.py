import pandas as pd
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

def batch_summarize_news(news_bodies, model, tokenizer, device, batch_size=32):
    summaries = []
    total_batches = len(news_bodies) // batch_size + (len(news_bodies) % batch_size > 0)

    for i in tqdm(range(0, len(news_bodies), batch_size), total=total_batches, desc="Batch Processing"):
        batch_news = news_bodies[i:i+batch_size]

        # Prepare model inputs using the updated tokenizer call
        inputs = tokenizer(
            ["summarize: " + news_body for news_body in batch_news],
            truncation=True,
            padding='longest',
            return_tensors="pt"
        ).to(device)

        # Generate summaries
        summary_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=4,
            length_penalty=2.0,
            max_length=60,
            min_length=12,
            early_stopping=True
        )

        # Decode the summaries
        batch_summaries = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        summaries.extend(batch_summaries)

    return summaries

def read_csv(filename):
    return pd.read_csv(filename)

def summarize_csv_file(input_csv_file, output_csv_file, model, tokenizer, device, batch_size=32):
    # Read the CSV file
    df = read_csv(input_csv_file).head(10)
    news_bodies = df['Body'].tolist()

    # Batch summarization
    summaries = batch_summarize_news(news_bodies, model, tokenizer, device, batch_size=batch_size)

    # Append summary column
    df['summary'] = summaries

    # Save the new CSV
    df.to_csv(output_csv_file, index=False)
    print(f"Summaries saved to: {output_csv_file}")

if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum').to('cuda:1')

    # Input and output file paths
    train_csv_file = 'FNC_Bin_Mix_Train.csv'
    test_csv_file = 'FNC_Bin_Mix_Test.csv'
    dev_csv_file = 'FNC_Bin_Mix_Dev.csv'

    output_train_csv_file = 'pegasus_train_summaries.csv'
    output_test_csv_file = 'pegasus_test_summaries.csv'
    output_dev_csv_file = 'pegasus_dev_summaries.csv'

    # Run summarization
    summarize_csv_file(dev_csv_file, output_dev_csv_file, model, tokenizer, 'cuda:1')
    summarize_csv_file(train_csv_file, output_train_csv_file, model, tokenizer, 'cuda:1')
    summarize_csv_file(test_csv_file, output_test_csv_file, model, tokenizer, 'cuda:1')
