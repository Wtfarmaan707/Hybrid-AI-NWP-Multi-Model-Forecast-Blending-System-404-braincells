# api/index.py
from flask import Flask # or FastAPI

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Vercel!"
