# Recurrent Neural Network

### This directory contains the implementation of RNN.
**Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory."**    
**They can read inputs 𝑥⟨𝑡⟩ (such as words) one at a time, and remember some contextual information through the hidden layer activations that get passed from one time step to the next.**     
**This allows a unidirectional (one-way) RNN to take information from the past to process later inputs. A bidirectional (two-way) RNN can take context from both the past and the future.**      
**It includes implementations of standard RNN blocks, GRUs, LSTMs (better at addressing vanishing gradients and more able to remember information from deep layers).**

## `Structure of the directory:`
  * `scratch.py` -> Directory with datasets used in the applications.
  * `utils.py` -> Implementation with 85%(97% train) accuracy on [Kaggle](https://www.kaggle.com/datasets/iarunava/happy-house-dataset/code) dataset using TF Keras' Sequential API.

**NOTE: For more details about implementation check comments.**

Copyright 2024 Vasile Alexandru-Gabriel (vasilealexandru37@gmail.com)