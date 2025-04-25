from transformers import pipeline
import gradio as gr
import os

os.environ["OMP_NUM_THREADS"] = "1"

model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1
)

def analyze(text):
    if not text.strip():
        return {"Error": "Ingresa texto válido"}
    
    try:
        result = model(text)[0]
        stars = int(result['label'][0])  # Extrae el número de estrellas (1-5)
        sentiment = "POSITIVO" if stars >= 4 else "NEUTRO" if stars == 3 else "NEGATIVO"
        return {
            "Estrellas": result['label'],
            "Sentimiento": sentiment,
            "Confianza": f"{result['score']:.2%}"
        }
    except Exception as e:
        return {"Error": str(e)}

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Opinión en español"),
    outputs=gr.JSON(),
    examples=[["El producto es excelente, lo recomiendo!"], 
              ["No cumple con lo prometido"]],
    title="Analizador para Opiniones en Español de Pol Monsalvo"
)

iface.launch(server_port=7860)