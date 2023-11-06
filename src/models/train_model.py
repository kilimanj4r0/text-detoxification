import warnings
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # Adjust
import torch

from datasets import Dataset, DatasetDict, load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Set seeds
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
warnings.filterwarnings('ignore')

# Dataset convertion to HF format

tokenizer_checkpoint_path = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_checkpoint_path)

train_df = pd.read_csv('../../data/interim/train.csv', index_col=0)
val_df = pd.read_csv('../../data/interim/val.csv', index_col=0)
test_df = pd.read_csv('../../data/interim/test.csv', index_col=0)


prefix = "detoxify:"
max_input_length = 256
max_target_length = 256


def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples["reference"]]
    targets = [ex for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)
ds = DatasetDict()
ds['train'] = train_ds
ds['val'] = val_ds
ds['test'] = test_ds
ds = ds.map(preprocess_function, batched=True)

# Training

model_name = 't5-base'
batch_size = 64
training_model_name = f"{model_name}-finetuned-paranmt500k-detox"

args = Seq2SeqTrainingArguments(
    output_dir=f'../models/{training_model_name}',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to='wandb',
)

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=ds['train'],
    eval_dataset=ds['val'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
