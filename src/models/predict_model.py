from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import pandas as pd

tokenizer_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

trained_model_name = f"../models/{tokenizer_name}-finetuned-paranmt500k-detox/checkpoint-2500"
model = T5ForConditionalGeneration.from_pretrained(trained_model_name)

# checkpoint_path = "PASTE HERE PATH TO DOWNLOADED CHEKPOINT"
# model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def detoxify_inference(sentences, prefix="detoxify:", top_k=120, max_length=256):
    outputs = []
    for sentence in tqdm(sentences):
        text = prefix + sentence + " </s>"

        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1,
        )
        for output in model_output:
            generated_sent = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            outputs.append(generated_sent)
    return outputs


test_data_path = '../data/interim/test.csv'
test_df = pd.read_csv(test_data_path, index_col=0)


n_samples = 25
sampled_test = test_df.sample(n=n_samples, random_state=111)
predicitons = detoxify_inference(sampled_test['reference'])

for ref, trn, pred in zip(sampled_test['reference'], sampled_test['translation'], predicitons):
    print(f'INIT: {ref}')
    print(f'GOLD: {trn}')
    print(f'PRED: {pred}')
    print('------------------------')