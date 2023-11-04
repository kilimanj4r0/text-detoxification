### Solution Bulding Report

# Baseline: Dictionary based

This solution was based on building a toxic dictionary with the help of text roberta toxicity classifier. It resluted in low quality of paraphrasing toxic text because the dictionary has the small size (we need significantly more time to increase its size) and this method is bad in preserving context since we have one-to-one correspondense between words. I have obtained 57.02% successfully turned into neutral text samples from the part of the data (filtered by similar texts). There are many ways to make this solution better.

# Hypothesis 1: Custom embeddings
...
# Hypothesis 2: More rnn layers
...
# Hypothesis 3: Pretrained embeddings
...
# Results
...