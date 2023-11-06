# Final Solution Report

## Introduction

For the Text Detoxification problem we were given a ParaNMT dataset with 500K samples of texts toxic and non-toxic texts with their charactersitics. The task was to build a model that can turn toxic text into a neutral, non-toxic one preserving the text context.

## Data analysis

Pandas and matplotlib were used to make a proper data analysis. It turns out that given dataset has issues with swapped samples (their toxicity levels were swapped). Data was checked for NaNs, maximum lenghts of texts were computed, and swapping was applied so that `reference` text always had toxicity level higher than `translation`. The plot of sample toxicities before and after preprocessing are provided below:

![](./figures/toxicities_before.png)
![](./figures/toxicities_after.png)

Data was then splitted into train, val, test splits. The number of samples correspondingly: (506998, 56334, 14445).

## Model Specification

Text-To-Text Transfer Transformer (T5) model is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.

The [base version](https://huggingface.co/t5-base) of T5 model have 223M parameters. It can be trained as a Seq2Seq model using HuggingFace `transformers` library.

**Note**: in few-shot learning of Mistral 7B were used 10 randomly sampled examples from train data split. They were fed into designed prompt with instrucitons to make the model inference.  

## Training Process

The [base version](https://huggingface.co/t5-base) (with 223M parameters) of T5 model was fine-tuned for 1 epoch with 64 batch size and sequence of 256 tokens maximum length. [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) was used to computed the loss. Checkpoint saving was also applied.

**Note**: few-shot learning of LLMs do not even need the training. Prompt design and intelligent output parser will make an inference for us immediately.  

## Evaluation

To evaluate all methods that were used I implemented evaluation of metrics:

1. [Style Transfer Accuracy](https://github.com/s-nlp/detox/blob/0ebaeab817957bb5463819bec7fa4ed3de9a26ee/emnlp2021/metric/metric.py#L27) — computes average score from [Roberta Toxicity Classifier by s-nlp](https://huggingface.co/s-nlp/roberta_toxicity_classifier).
2. [BLEU Score](https://huggingface.co/spaces/evaluate-metric/bleu) — compares predicted text with translations, so that "the closer a predicted text is to a translation, the better it is".

Also, the results of models performance were accessed manually for sanity check and training improvement.

## Results

| Model | Style Transfer Accuracy | BLEU Score | 
| ----- | ------- | ---------- |
| dictionary-based | out of comparison | out of comparison |  
| fine-tuned t5-base | 72.83%  | **43.08%** | 
| few-shot mistral7b | **82.78%** | 29.51% |

Even though, Fine-tuned T5-base outperforms Few-shot Mistral 7B in BLEU, the LLM is better in Style Transfer. I think we have such results because T5 is more robust model for seq2seq translation task (and results can be even improved taking larger versions), while LLM is tuned to follow specific instructions so it better keeps the style.
