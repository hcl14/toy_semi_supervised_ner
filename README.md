# Toy semi-supervised NER
-------------------------


Here I experiment just for fun, creating semi-supervised method to perform training of NER model without labeled training data available - an approach, which can be used for training NER classifiers for specific text corpora. All that is needed is as much as possible examples of named entities which will occur in those data.<br>
The goal is to deal with the problem of the lack of supervised data for training NER models. Two most known datasets are CONLL2003 (~4 MB) and WNUT-17 (2.3 mb) [1](https://github.com/davidsbatista/NER-datasets). Also, which is more important, it is almost impossible to find supervised training data for specific domain (medical, twitter, etc). Using Amazon Mechanical Turk authors of the paper Annotating Large Email Datasets for Named Entity Recognition with Mechanical Turk [2](http://www.aclweb.org/anthology/W10-0712) managed to collect 8mb dataset after 4 months, thus creating own dataset is long and expensive.<br>
The main assumption behind the usage of such small datasets is that word embeddings should capture contextual information, so the features indicating particular word being an enitity should be somehow encoded in the embeddings. Thus training model on a small dataset is just a sort of a training new simple classifier over existing embeddings and should not require much data. However, perhaps existing 3-8mb datasets are insufiicient for capturing all the information. <br>

The idea of this toy model is the following: all the named entities of some type (`Company` only at the moment) should have something in common in the context where they appear, so even the one who does not know such a company, may deduce that this is a company name in text. Thus, by providing lists of companies, persons, etc I try to create context embeddings using surrounding words in the chosen corpora, and train classifier for them. After that, another named entity, which did not occur during training, should be correctly classified by its context in this text domain. <br>


Training data is scraped from Wikipedia using the list of companies. Article about the company is being scanned, sentences containing company name are split into two parts which are being fed into two biLSTMs with summarizing layer above them. To feed data into the model, I use GoogleNews word2vec embeddings [3](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). Zero embeddings are used for unknown words. Please edit the code in `train.py` and set path. Negative examples are created by splitting the sentences, which does not contain company name, at random word. The resulting training dataset is about 11 MB.

## Requirements

Python 3, Keras framework.


## Output

There is no need to export toy model in `train.py`, as it creates limited vocabulary matrix from the dataset. The quality is of course very low, but this idea of context capturing can be used as a part for something bigger. You can train it in the python shell and play a bit by evaluating different sentences:

`$python3 -i train.py`

```
Loading word2vec model
Loading datafiles
positive examples: 17503
negative examples: 56105
Building vocabulary
Constructing embedding matrix
Preparing model inputs
vocab_size: 57393
max_len: 15

Train on 66247 samples, validate on 7361 samples
Epoch 1/5
66247/66247 [==============================] - 122s 2ms/step - loss: 0.3649 - acc: 0.8486 - val_loss: 0.3208 - val_acc: 0.8730

Epoch 00001: val_loss improved from inf to 0.32082, saving model to models/model-best.hdf5
Epoch 2/5
66247/66247 [==============================] - 120s 2ms/step - loss: 0.3011 - acc: 0.8771 - val_loss: 0.2875 - val_acc: 0.8859

Epoch 00002: val_loss improved from 0.32082 to 0.28752, saving model to models/model-best.hdf5
Epoch 3/5
66247/66247 [==============================] - 123s 2ms/step - loss: 0.2568 - acc: 0.8980 - val_loss: 0.2816 - val_acc: 0.8833

Epoch 00003: val_loss improved from 0.28752 to 0.28160, saving model to models/model-best.hdf5
Epoch 4/5
66247/66247 [==============================] - 123s 2ms/step - loss: 0.2144 - acc: 0.9169 - val_loss: 0.2758 - val_acc: 0.8943

Epoch 00004: val_loss improved from 0.28160 to 0.27575, saving model to models/model-best.hdf5
Epoch 5/5
66247/66247 [==============================] - 122s 2ms/step - loss: 0.1721 - acc: 0.9354 - val_loss: 0.2821 - val_acc: 0.8905

Epoch 00005: val_loss did not improve from 0.27575

>>> ner(['we','sued','THECORP','for','million','dollars'])
we(True)
sued(False)
THECORP(True)
for(False)
million(False)
dollars(False)

```

