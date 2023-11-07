from transformers import pipeline
from pathlib import Path
import time

here = Path(__file__).parent

classifier = pipeline("text2text-generation", model=here / "./model")

clock = time.time()
print(classifier("generate recipe from ingredients:steak, oil, butter, garlic, rosemary"))
print("Time elapsed: " + str(time.time() - clock))
