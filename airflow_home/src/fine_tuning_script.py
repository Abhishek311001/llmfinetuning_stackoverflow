from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd

preprocessed_data = pd.read_parquet("/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow/data/preprocessed/preprocessed_data.parquet")

print(preprocessed_data.head())
