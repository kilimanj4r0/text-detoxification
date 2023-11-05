# Text Detoxification Assignment

🧑‍💻 **Vladimir Makharev**
📧 v.makharev@innopolis.university
🎓 B20-AI-01

# How to use

### Preresquities

```bash
pip install -r requirements.txt
```

If you want reproduce Few-shot learning results with Mistral 7B:

1. [Download ollama](https://ollama.ai/download), then:

```bash
ollama serve
ollama pull mistral
```

## T5 fine-tuning on ParaNMT-500K detoxification dataset

```bash
cd src
```

### Prepare data

```bash
python data/make_dataset.py
```

### Train model

```bash
python models/train_model.py
```

### Predict

```bash
python models/predict_model.py
```

## Appendix

### Visualisations

```bash
python visualization/visualize.py
```