# Text Detoxification Assignment

🧑‍💻 **Vladimir Makharev**
📧 v.makharev@innopolis.university
🎓 B20-AI-01

# How to use

The ParaNMT-500K (filtered) detoxification dataset was used and can be downloaded [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).

### Preresquities and data preparaion

```bash
pip install -r requirements.txt
python src/data/make_dataset.py
```

## Few-shot learning Mistral 7B

1. [Download ollama](https://ollama.ai/download), then:

```bash
ollama serve
ollama pull mistral
```

2. Use [LLM notebook](https://github.com/kilimanj4r0/text-detoxification/blob/main/notebooks/2.2-llm.ipynb).

## Fine-tuning T5-base

1. Download [model checkpoint](https://disk.yandex.ru/d/xPIno6SPXFL7dA) (2.28GB)
2. Paste local path to checkpoint to `src/models/predict_model.py`, then:

```bash
cd src
python models/predict_model.py
```

2. (Alternative) to train from scratch:

```bash
cd src
python models/train_model.py
python models/predict_model.py
```

## Appendix

### Reproduce visualisations

```bash
python visualization/visualize.py
```
