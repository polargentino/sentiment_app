from transformers import pipeline
import tkinter as tk
from tkinter import ttk

# 1. Cargar el modelo (se descarga automáticamente la primera vez)
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Función para analizar texto
def analyze_sentiment():
    text = text_entry.get("1.0", "end-1c")  # Obtener texto del cuadro
    if text.strip():  # Si no está vacío
        result = model(text)[0]  # Hacer la predicción
        sentiment = result['label']
        confidence = result['score']
        # Mostrar resultado con emojis y colores
        if sentiment == "POSITIVE":
            output.config(text=f"✅ Positivo (Confianza: {confidence:.2%})", foreground="green")
        else:
            output.config(text=f"❌ Negativo (Confianza: {confidence:.2%})", foreground="red")

# 3. Configurar la interfaz gráfica
root = tk.Tk()
root.title("Analizador de Sentimientos - By Compaq CQ40 - Pol Monsalvo")
root.geometry("500x300")

# Widgets
label = ttk.Label(root, text="Ingresa tu texto:")
text_entry = tk.Text(root, height=10, width=60)
analyze_btn = ttk.Button(root, text="Analizar Sentimiento", command=analyze_sentiment)
output = ttk.Label(root, text="", font=('Helvetica', 12))

# Diseño
label.pack(pady=5)
text_entry.pack(pady=5)
analyze_btn.pack(pady=10)
output.pack(pady=10)

root.mainloop()
