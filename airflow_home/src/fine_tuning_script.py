from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import sys

logger = logging.getLogger("Fine Tuning Logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


logger.info("Reading preprocessed data....")
preprocessed_data = pd.read_parquet("/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow/data/preprocessed/preprocessed_data.parquet")
logger.info("Reading preprocessed data complete!!")
logger.info("train test val split")
preprocessed_data["score"] = preprocessed_data["score"].astype(int)
train_df, temp_df = train_test_split(preprocessed_data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df["labels"] = train_df["score"].apply(lambda x: 0 if x <= 0 else 1)
test_df["labels"] = test_df["score"].apply(lambda x: 0 if x <= 0 else 1)
val_df["labels"] = val_df["score"].apply(lambda x: 0 if x <= 0 else 1)

# print(len(train_df[train_df["score"] <= 0]), len(train_df[train_df["score"] > 0], len(train_df[train_df["score"] > 5])))
train_dataframe = Dataset.from_pandas(train_df)
test_dataframe = Dataset.from_pandas(test_df)
val_dataframe = Dataset.from_pandas(val_df)

logger.info("fetching model and tokenizers")

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,                # Low-rank dimension. Controls model capacity added by LoRA adapters.
    lora_alpha=16,      # Scaling factor for LoRA updates.
    target_modules=["q_lin", "v_lin"],  # Parts of the attention layers to inject LoRA modules into.
    lora_dropout=0.05,  # Dropout rate to regularize training.
    bias="none",        # No bias adjustment in LoRA layers.
    task_type=TaskType.SEQ_CLS,  # Task type: sequence classification.
)

logger.info("Fine Tuning...")

model = get_peft_model(model, lora_config)

def preprocess(batch):
    # Tokenize question and answer text inputs
    tokenized_batch = tokenizer(
        batch["Question"], batch["Answer"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    return tokenized_batch

val_dataset = val_dataframe.map(preprocess, batched=True)
train_dataset = train_dataframe.map(preprocess, batched=True)
test_dataset = test_dataframe.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow/airflow_home/fine_tuning_output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # You'll define this metric below
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",  # or "mlflow" if using tracking
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
logger.info("Training")

trainer.train()

logger.info("Saving model and trainer")
model.save_pretrained("/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow/airflow_home/Finetuning_models_trainers/distilbert_peft_lora")
tokenizer.save_pretrained("/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow/airflow_home/Finetuning_models_trainers/distilbert_peft_lora")

logger.info("Evaluating")
results = trainer.evaluate(test_dataset)
print("Test results:", results)
