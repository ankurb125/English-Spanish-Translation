# English-Spanish-Translation

1. Goal: Translation of English to Spanish is done using mT5 Transformer.
2. The Europarl dataset is used for fine-tuning the mT5 model.
3. Pre-processing was done using the Moses decoder script.
4. Evaluated the translations using BLEU, ChrF, and TER metrics.



# Download the model
https://huggingface.co/ankurb125/ankur-mt5-small-finetuned-en-to-es

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="ankurb125/ankur-mt5-small-finetuned-en-to-es")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ankurb125/ankur-mt5-small-finetuned-en-to-es")

model = AutoModelForSeq2SeqLM.from_pretrained("ankurb125/ankur-mt5-small-finetuned-en-to-es")
