# Language Modeling with Smoothing
N-gram language model with popular smoothing methods integrated (i.e. laplace/held-out/cross-validation smoothing,)

## Features
- Support n-gram language modeling (n is a tunable parameter, you may set n to 2 for bigram, 3 for trigram, etc.)
- Support popular smoothing methods, including Laplace Smoothing, Held-out Smoothing and Cross-validation Smoothing
- Support perplexity computation on a given test corpus 
- Support sentence generation
- Support Spearman's rank correlation coefficient computation among different different smoothed LMs


## Tutorial
1. Install package dependencies
```
pip install requirements.txt
```

2. Go ahead and run the model!
```
python langauge_model.py -data ./corpus/ -n 2 -method l -num 5
```
where `-data` represents the path to the corpus, `n` represents the parameter for tuning n-gram, `-method` represents the smoothing method to be used (`l` for laplace smoothing, `h` for held-out smoothing, `c` for cross-validation smoothing), `-num` represents the number of sentences to be generated.
3. For next token prediction and Spearman's rank correlation coefficient computation, please refer to the code for details.

## Remarks
- For detailed description of the smoothing methods used in this project, please refer to </br>Manning, Christopher, and Hinrich Schutze. *Foundations of statistical natural language processing.* MIT press, 1999. ([link](https://www.cs.vassar.edu/~cs366/docs/Manning_Schuetze_StatisticalNLP.pdf))
- This project assumes the corpus has been segmented. Otherwise please conduct text segmentation on your corpus before running this program. For Chinese text segmentation, you may consider using [jieba](https://github.com/fxsjy/jieba).
