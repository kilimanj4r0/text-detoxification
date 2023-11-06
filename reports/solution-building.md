# Solution Bulding Report

To evaluate all methods that were used I implemented evaluation of metrics:

1. Style Transfer Accuracy
2. BLEU score

Also, the results of models performance were accessed manually.

## Baseline: Dictionary based

This solution was based on building a toxic dictionary with the help of text roberta toxicity classifier. It resluted in low quality of paraphrasing toxic text because the dictionary has the small size (we need significantly more time to increase its size) and this method is bad in preserving context since we have one-to-one correspondense between words. I have obtained 57.02% successfully turned into neutral text samples from the part of the data (filtered by similar texts). There are many ways to make this solution better.

## Hypothesis 1: Attention is all we need

This hypothesis is simple — just use state-of-the-art Transformer architecture. The authors used [fine-tuned on pharaphrase T5 base model](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis) to fine-tune on the ParaNMT 500K dataset ([their work](https://huggingface.co/s-nlp/t5-paranmt-detox)). So, I decided to make one step back and fine-tune T5 model. For loss metric so-called serious BLEU score was chosen, namely, [SacreBLEU](https://github.com/mjpost/sacreBLEU). This method shows promising results: Style Transfer Accuracy and  BLEU score.

## Hypothesis 2: LLMs power

With rise of open source LLM models, it became easy to implement few-shot learning of LLMs. I decided to use LangChain and Ollama frameworks to build a few-shot prompting approach on Mistral 7B. This model is instruct model with 7.3B parameters that outperforms Llama 2 13B. Thanks Ollama for uncensored and easy-to-run API. This approach requires prompt (instruction) design, showing some examples to model and well-parsed output of the model. After all, applying 10-shot learning to Mistral 7B results in 29.51% BLEU score and 82.78% Style Transfer Accuracy, which is quite good for such setting. The pros of such method is that it does not require any training, while the cons of it that it requires a lot of memory and inference might take longer time than for Transformer-based models.

# Results

Attempt of using LLM was a good journey and I got interesting comparable results. Of course, other scoring metrics can be added to evaluate models more properly. It is hard to compare dictionary based method to others since it has really poor performance. Therefore, METHOD performed better than all others.