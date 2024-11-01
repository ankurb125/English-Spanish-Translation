# -*- coding: utf-8 -*-
"""mt5_with_europarl_evaluation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GodpzR6K4zbqhkv4tZX4yYaYXXuGG2yU
"""

!pip install datasets evaluate transformers[sentencepiece]
from transformers import pipeline

!pip install sacrebleu

# from google.colab import drive
# drive.mount('/content/drive')
# from huggingface_hub import notebook_login

# notebook_login()

# from huggingface_hub import Repository, get_full_repo_name

# model_name = "ankur-mt5-small-finetuned-en-to-es"
# repo_name = get_full_repo_name(model_name)
# repo_name

# output_dir = "ankur-mt5-small-finetuned-en-to-es"
# repo = Repository(output_dir, clone_from=repo_name)

model_checkpoint = "ankurb125/ankur-mt5-small-finetuned-en-to-es"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")

c = translator('Hi I am Ankur, I will be working at OMU as an internship student.', max_length=400)[0]['translation_text']
c

a = translator('My name is Ankur Bhatt, pursuing masters in computer science specializing in Artificial Intelligence. I have been working with DFKI GmbH since January 2022 and I have a huge interest in NLP.',max_length=400)[0]['translation_text']

b = translator('Hi, How are you all doing? Today, we will start with this course this semester.', max_length=400)[0]['translation_text']
a,b

import evaluate

metric = evaluate.load("sacrebleu")
predictions = [a,b]
references = ['Mi nombre es Ankur Bhatt, estoy cursando una maestría en informática con especialización en Inteligencia Artificial. Trabajo con DFKI GmbH desde enero de 2022 y tengo un gran interés en la PNL.','Hola, ¿cómo están todos? Hoy, vamos a empezar con este curso este semestre.']
metric.compute(predictions=predictions, references=references)

translator('Yes, it is possible to check how a person\'s self-confidence is affected by providing them with visual information, such as a video, and then asking them questions related to the content of the video.Self-confidence can be defined as a belief in one\'s abilities, qualities, and judgment. When someone is confident, they tend to feel more positive and assured about their abilities, which can lead to better performance and outcomes.', max_length=400)

def load_doc(filename):
  file = open(filename, mode='rt', encoding='utf-8')
  text = file.read()
  file.close()
  return text
def to_sentences(doc):
  return doc.strip().split('\n')
def sentence_lengths(sentences):
  lengths = [len(s.split()) for s in sentences]
  return min(lengths), max(lengths)

# clean_en_filename = '../datasets/NLP_Data/clean.en'
clean_en_filename = 'drive/MyDrive/clean_newstest.en'
clean_en_doc = load_doc(clean_en_filename)
clean_en_sentences = to_sentences(clean_en_doc)
minlen, maxlen = sentence_lengths(clean_en_sentences)
print('English cleaned data: sentences=%d, min=%d, max=%d' % (len(clean_en_sentences), minlen, maxlen))

# clean_es_filename = '../datasets/NLP_Data/clean.es'
clean_es_filename = 'drive/MyDrive/clean_newstest.es'
clean_es_doc = load_doc(clean_es_filename)
clean_es_sentences = to_sentences(clean_es_doc)
minlen, maxlen = sentence_lengths(clean_es_sentences)
print('Spanish cleaned data: sentences=%d, min=%d, max=%d' % (len(clean_es_sentences), minlen, maxlen))

predictions =[]
c = 0
for i in range (0,len(clean_en_sentences)):
  t = translator(clean_en_sentences[i],max_length=200)
  t =t[0]['translation_text']
  c+=1
  print(c)
  predictions.append(t)

predictions[:10]

clean_es_sentences[:10]

references=clean_es_sentences[:len(clean_es_sentences)]

import evaluate

metric = evaluate.load("sacrebleu")
metric.compute(predictions=predictions, references=references)

metric = evaluate.load("bleu")
metric.compute(predictions=predictions, references=references)

metric = evaluate.load("chrf")
metric.compute(predictions=predictions, references=references)

metric = evaluate.load("ter")
metric.compute(predictions=predictions, references=references)

