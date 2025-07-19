import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from unsloth import FastLanguageModel
from huggingface_hub.utils._token import get_token


# ================ GPU Setup =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============== Load LLAMA Model =================
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth"
)

# =============== Dataset Class ===================
class PromptDataset(Dataset):
    def __init__(self, df):
        self.headlines = df["Headline"].tolist()
        self.bodies = df["Body"].tolist()
        self.labels = df["Label"].tolist()

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline = self.headlines[idx]
        body = self.bodies[idx]
        label = self.labels[idx]
        answer = "similar" if label == 0 else "unlike"

        prompt = f"""<|start_header_id|>user<|end_header_id|>

Headline: {headline}
Body: {body}

Are the headline and body similar or unlike? Respond with only "similar" or "unlike".

<|start_header_id|>assistant<|end_header_id|> {answer}"""

        tokenized = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokenized.items()}

# ============== Load Data ========================
train_df = pd.read_csv("train.csv").head(10)
dev_df = pd.read_csv("dev.csv").head(10)
test_df = pd.read_csv("test.csv").head(10)

train_loader = DataLoader(PromptDataset(train_df), batch_size=1, shuffle=True)
dev_loader = DataLoader(PromptDataset(dev_df), batch_size=1, shuffle=False)
test_loader = DataLoader(PromptDataset(test_df), batch_size=1, shuffle=False)

# ============== Optimizer ========================
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * 3
)

# ============== Train, Validate, Test ================
epochs = 1
model.train()

for epoch in range(epochs):
    total_loss = 0.0

    # === Training ===
    print(f"\n🟢 Epoch {epoch+1}/{epochs} - Training")
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"🔵 Avg Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    print(f"🟡 Epoch {epoch+1} - Validation")
    model.eval()
    dev_preds = []
    for batch in tqdm(dev_loader, desc="Validation"):
        with torch.no_grad():
            input_texts = []
            for i in range(len(batch["input_ids"])):
                decoded = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                input_texts.append(decoded)

            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                max_new_tokens=1,
                do_sample=False
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            dev_preds.extend(predictions)

    # Save dev predictions
    pd.DataFrame({"Prediction": dev_preds}).to_csv(f"dev_predictions_epoch_{epoch+1}.csv", index=False)
    print(f" Dev predictions saved to dev_predictions_epoch_{epoch+1}.csv")

    # === Test ===
    print(f"🔴 Epoch {epoch+1} - Test")
    test_preds = []
    for batch in tqdm(test_loader, desc="Testing"):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                max_new_tokens=1,
                do_sample=False
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            test_preds.extend(predictions)

    # Save test predictions
    pd.DataFrame({"Prediction": test_preds}).to_csv(f"test_predictions_epoch_{epoch+1}.csv", index=False)
    print(f" Test predictions saved to test_predictions_epoch_{epoch+1}.csv")

# ============ Save Final Model ==================
model.save_pretrained("llama3-similarity-lora")
tokenizer.save_pretrained("llama3-similarity-lora")
print(" Final model saved to llama3-similarity-lora")
