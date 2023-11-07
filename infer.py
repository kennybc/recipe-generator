from transformers import pipeline
from pathlib import Path
import time
from transformers import logging, AutoTokenizer, AutoModelForSeq2SeqLM

here = Path(__file__).parent

tokenizer = AutoTokenizer.from_pretrained(here/"./model", model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(here/"./model")

text = "generate recipe from ingredients: chicken, tortillas, cheese, butter, peppers, onion"
tokenized = tokenizer(
    text, max_length=1024, truncation=True, return_tensors="pt")

generation_kwargs = {
    "max_length": 100000,
    "min_length": 64,
    "do_sample": True,
}

clock = time.time()
output = model.generate(
    input_ids=tokenized["input_ids"],
    attention_mask=tokenized["attention_mask"],
    **generation_kwargs
)
print(tokenizer.decode(output[0], skip_special_tokens=False))
print("Time elapsed: " + str(time.time() - clock))
