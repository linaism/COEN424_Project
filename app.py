import gradio as gr
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for

from omnixai.data.tabular import Tabular

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello"

if __name__ == '__main__':
    # Start the Gradio interface
    app.run(host='0.0.0.0', port=80)
