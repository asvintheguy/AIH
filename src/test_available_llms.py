from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
  api_key="AIzaSyDg0uPj-GOtVoCH2M2M5ON1UnR-yjZLip8",
)

models = client.models.list()
for model in models:
  print(model.id)
