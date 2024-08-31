---
layout: post
title:  "An implementation guide to Word2Vec using NumPy and Google Sheets!"
date:   2018-12-06 00:00:00 +0800
# categories: main
---
Word2Vec is touted as one of the biggest, most recent breakthrough in the field of Natural Language Processing (NLP). The concept is simple, elegant and (relatively) easy to grasp. A quick Google search returns multiple results on how to use them with standard libraries such as [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) and [TensorFlow](https://www.tensorflow.org/tutorials/representation/word2vec). Also, for the curious minds, check out the original implementation using C by [Tomas Mikolov](https://github.com/tmikolov/word2vec). The original paper can be found [here](https://arxiv.org/pdf/1301.3781.pdf) too.

The main focus on this article is to present Word2Vec in detail. For that, I implemented Word2Vec on Python using NumPy (with much help from other tutorials) and also prepared a Google Sheet to showcase the calculations. Here are the links to the [code](https://github.com/DerekChia/word2vec_numpy) and [Google Sheet](https://docs.google.com/spreadsheets/u/3/d/1mgf82Ue7MmQixMm2ZqnT1oWUucj6pEcd2wDs_JgHmco/edit).

![Image]({{ site.baseurl }}/assets/images/word2vec-gsheets1.png){: width="100%" }

### Intuition

The objective of Word2Vec is to generate vector representations of words that carry semantic meanings for further NLP tasks. Each word vector is typically several hundred dimensions and each unique word in the corpus is assigned a vector in the space. For example, the word “happy” can be represented as a vector of 4 dimensions [0.24, 0.45, 0.11, 0.49] and “sad” has a vector of [0.88, 0.78, 0.45, 0.91].

The transformation from words to vectors is also known as [word embedding](https://en.wikipedia.org/wiki/Word_embedding). The reason for this transformation is so that machine learning algorithm can perform linear algebra operations on numbers (in vectors) instead of words.

To implement Word2Vec, there are two flavors to choose from — __Continuous Bag-Of-Words (CBOW)__ or __continuous Skip-gram (SG)__. In short, CBOW attempts to guess the output (target word) from its neighbouring words (context words) whereas continuous Skip-Gram guesses the context words from a target word. Effectively, Word2Vec is based on [distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics) where the context for each word is in its nearby words. Hence, by looking at its neighbouring words, we can attempt to predict the target word.

According to Mikolov (quoted in [this](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures) article), here is the difference between Skip-gram and CBOW:

> Skip-gram: works well with small amount of the training data, represents well even rare words or phrases

> CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words

To elaborate further, since __Skip-gram__ learns to predict the context words from a given word, in case where two words (one appearing infrequently and the other more frequently) are placed side-by-side, both will have the same treatment when it comes to minimising loss since each word will be treated as both the target word and context word. Comparing that to __CBOW__, the infrequent word will only be part of a collection of context words used to predict the target word. Therefore, the model will assign the infrequent word a low probability.

![Image]({{ site.baseurl }}/assets/images/word2vec-gsheets2.png){: width="100%" }

### Implementation Process
In this article, we will be implementing the __Skip-gram__ architecture. The content is broken down into the following parts for easy reading:

1. __Data Preparation__ — Define corpus, clean, normalise and tokenise words
2. __Hyperparameters__ — Learning rate, epochs, window size, embedding size
3. __Generate Training Data__ — Build vocabulary, one-hot encoding for words, build dictionaries that map id to word and vice versa
4. __Model Training__ — Pass encoded words through forward pass, calculate error rate, adjust weights using backpropagation and compute loss
5. __Inference__ — Get word vector and find similar words
6. __Further improvements__ — Speeding up training time with Skip-gram Negative Sampling (SGNS) and Hierarchical Softmax


