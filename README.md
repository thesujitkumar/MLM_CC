## Dependencies

* python3
* pytorch
* tqdm
* sklearn
* pandas
* openpyxl
* torch-geometric 2.5.3


## Model Training and Testing
 **To Run the code, kindly follow step-by-step instructions:**
 1. Extract Dataset from source reported in the paper.
 2. Prepare train.csv, test.csv, dev.csv.
**For Finetuning Transformer-based Model:**
1. Switch to BERT Finetune/ Folder
2. Run python BERT.py
3. Run python RoBERTa.py

**For Cascade Classification Models:**
1. Switch to Cascade_Classification/ Folder
2. First, Generate Summary of News Body.
     (i) Pegasus:  python Pegasus_news_body_summary.py
     (ii) T5:      python T5_news_body_summary.py
     (iii) Gemini: python gemini_news_body_summary.py
4. Run python CC_RoBERTa.py

**For Masked Language Modelling:**
1.  Switch to MLM/ Folder
2.  BERT:  BERT_prompt_Tuning.py
3.  Llama: Lama_MLM_Prompt.py


## Contact
sujitkumar@iitg.ac.in, kumar.sujit474@gmail.com
