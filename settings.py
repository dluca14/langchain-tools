import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(f'{BASE_DIR}/.env')

os.environ["FIREWORKS_API_KEY"] = os.getenv('FIREWORKS_API_KEY', default='')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', default='')
os.environ["NOMIC_API_KEY"] = os.getenv('NOMIC_API_KEY', default='')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY', default='')
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = 'default'
