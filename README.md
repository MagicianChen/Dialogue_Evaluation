# Dialogue_Evaluation
Several evaluation metric for dialogue generation task. In this project, i conclude seven evaluation metrics and record the demo here.
## Metric list
1. BLEU: calculate the word character matching degree. 
2. Distinct: measure the diversity of the output
3. Embedding-based metrics(Average, Extrema, Greedy): measure the semantic similarity between reference and model output
4. Coherence: measure how similarity between dialogue context and model output response.

## Requirement
```
python == 3.x
nltk  
ipdb  
fasttext  
```
The fasttext can be replaced by other word embedding model like Glove or word2vec. You need to download the pre-trained model before running the program. It's better to use pre-trained model on all datasets for fair.

## Usage
you can run the metric.py and the program will output the result in a txt file. Since there are several metrics based on word embedding model, it will cost much more time.
```
python metric.py  
```

## Reference
1. [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)
2. [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
3. [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/pdf/1603.08023.pdf?__hstc=36392319.57feae9086cbe66baa94bf32ef453412.1482451200081.1482451200082.1482451200083.1&__hssc=36392319.1.1482451200084&__hsfp=528229161)
4. [Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity](https://arxiv.org/pdf/1809.06873.pdf?source=post_page---------------------------)

