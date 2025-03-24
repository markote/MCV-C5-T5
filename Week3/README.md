# Week 3

## Paper

[Overleaf Project (read-only)](https://overleaf.cvc.uab.es/read/frjjcrdqycyv#fc112e)

## Presentation

[Google Slides](https://docs.google.com/presentation/d/1yXmUeakQ18B4oaQmoMKoDUiq995nDZLAs4WuIH6YhRw/edit?usp=sharing)

## Code

* Install requirements with:
```
pip install r requirements.txt
```

* Data split script, for splitting and initial data filtering:

```
python data_splitter.py
```


* The additional filtering is found [here](./FilterBookCoverImage.ipynb)

* The initial model can be run like this:
```
python initial_model.py
```

* Vocab corpus generator for word tokens:
```
python word_tokenizer_recettes_corpus.py
```

* For testing and training models you can use the following scripts for each word level piece:
```
python bert_wordpiece_model.py
python roberta_wordpiece_model.py
python word_model.py
python model_morelayer.py
```
