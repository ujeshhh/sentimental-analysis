import gradio as gr
from transformers import pipeline

# Load multilingual sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"{result['label']} ({round(result['score'], 2)})"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text"),
    outputs="text",
    title="Multilingual Sentiment Analysis 🌐",
    description="Analyze text sentiment in multiple languages using Hugging Face."
)

if __name__ == "__main__":
    iface.launch()
