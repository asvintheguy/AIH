from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

models = client.models.list()
for model in models:
  print(model.id)
