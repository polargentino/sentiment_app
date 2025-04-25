from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Configuración para optimizar rendimiento
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cargamos el modelo
model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1,
    truncation=True
)

def analyze(text):
    if not text.strip():
        return {"Error": "Ingresa texto válido"}, None
    
    try:
        # 1. Análisis de sentimiento
        result = model(text[:512])[0]
        stars = int(result['label'][0])
        sentiment = "POSITIVO" if stars >= 4 else "NEUTRO" if stars == 3 else "NEGATIVO"
        
        # 2. Detección de aspectos mejorada
        text_lower = text.lower()
        aspects = {
            "Rendimiento": any(word in text_lower for word in [
                "contento", "full", "usa", "funciona", "bien", "rápido", "fluido"
            ]),
            "Calidad-Precio": any(phrase in text_lower for phrase in [
                "barato", "económico", "lo vale", "precio", "coste"
            ]),
            "Completitud": not any(phrase in text_lower for phrase in [
                "no venía", "faltó", "sin ", "no incluye"
            ]),
            "Recomendación": any(word in text_lower for word in [
                "recomiendo", "recomendaría", "contento", "feliz"
            ])
        }
        
        # 3. Gráfico profesional mejorado
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Configuramos colores y estilo
        colors = ['#4CAF50' if val else '#F44336' for val in aspects.values()]
        aspect_names = list(aspects.keys())
        detected = list(aspects.values())
        
        # Creamos barras horizontales con efectos
        bars = ax.barh(aspect_names, [1]*4, color=colors, alpha=0.8, height=0.6)
        
        # Añadimos efectos de profundidad
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)
            bar.set_hatch('' if bar.get_facecolor() == '#4CAF50' else 'xxx')
        
        # Añadimos iconos y texto descriptivo
        for i, (name, is_detected) in enumerate(aspects.items()):
            if is_detected:
                ax.text(0.5, i, "✓ DETECTADO", 
                       va='center', ha='center', 
                       color='white', fontweight='bold',
                       fontsize=11, bbox=dict(facecolor='#2E7D32', alpha=0.9))
            else:
                ax.text(0.5, i, "NO DETECTADO", 
                       va='center', ha='center', 
                       color='white', fontweight='bold',
                       fontsize=10, bbox=dict(facecolor='#C62828', alpha=0.7))
        
        # Personalización avanzada
        ax.set_title('DETALLE DE ASPECTOS ANALIZADOS', 
                   pad=20, fontsize=14, fontweight='bold', color='#333333')
        
        # Quitamos ejes y bordes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Añadimos leyenda personalizada
        ax.text(1.02, 0.5, "🔍 Análisis realizado sobre:\n" + text[:100] + ("..." if len(text)>100 else ""),
               transform=ax.transAxes, va='center', fontsize=9,
               bbox=dict(facecolor='#f5f5f5', alpha=0.5))
        
        plt.tight_layout()
        plt.close(fig)
        
        # 4. Resultado final
        json_result = {
            "Resumen": {
                "Estrellas": result['label'],
                "Sentimiento": sentiment,
                "Confianza": f"{result['score']:.2%}",
                "Aspectos_Positivos": f"{sum(aspects.values())}/{len(aspects)}"
            },
            "Detalles": {
                "Texto_analizado": text[:200] + "..." if len(text) > 200 else text,
                "Aspectos": {k: "✅" if v else "❌" for k,v in aspects.items()}
            }
        }
        
        return json_result, fig
        
    except Exception as e:
        return {"Error": str(e)}, None

# Interfaz mejorada
iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="📝 Escribe tu reseña", 
        placeholder="Ej: Estoy muy contento con el producto...",
        lines=3
    ),
    outputs=[
        gr.JSON(label="📊 Resultado Completo"),
        gr.Plot(label="📌 Visualización de Aspectos")
    ],
    examples=[
        ["Muy contento con la pc mi hijo la usa full"],
        ["Buen producto pero no traía todos los accesorios"],
        ["No lo recomiendo, se calienta mucho y es caro"]
    ],
    title="🛒 Analizador Profesional de Reseñas",
    description="""Sistema avanzado que analiza sentimientos y detecta aspectos clave en reseñas de productos""",
    allow_flagging="never"
)

iface.launch(server_port=7860)