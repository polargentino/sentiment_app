from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# Configuración para optimizar rendimiento en CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cargamos el modelo de análisis de sentimientos
model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1,  # Fuerza uso de CPU
    truncation=True
)

# Paleta de colores personalizada
cmap = LinearSegmentedColormap.from_list("custom", ["#FF5252", "#FFEB3B", "#4CAF50"])

def analyze(text):
    if not text.strip():
        return {"Error": "Ingresa texto válido"}, None
    
    try:
        # 1. Análisis de sentimiento principal
        result = model(text[:512])[0]  # Limitamos a 512 tokens
        stars = int(result['label'][0])
        sentiment = "POSITIVO" if stars >= 4 else "NEUTRO" if stars == 3 else "NEGATIVO"
        
        # 2. Detección de aspectos mejorada
        text_lower = text.lower()
        aspects = {
            "Rendimiento": any(word in text_lower for word in [
                "excelente", "buen", "rápido", "fluido", "velocidad", "potente", 
                "satisfecho", "cumple", "anda de 10", "óptimo"
            ]),
            "Calidad-Precio": any(phrase in text_lower for phrase in [
                "calidadprecio", "buen precio", "relación calidad", "lo vale",
                "económico", "barato", "coste", "inversión", "precio-calidad"
            ]),
            "Completitud": not any(phrase in text_lower for phrase in [
                "no venía", "faltó", "sin cables", "no incluye", "carece",
                "no trae", "incompleto", "necesita comprar"
            ]),
            "Recomendación": any(word in text_lower for word in [
                "recomiendo", "recomendaría", "excelente compra", 
                "volvería a comprar", "lo elegiría"
            ])
        }
        
        # 3. Gráfico avanzado de aspectos
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Calculamos puntuación general (para el gradiente de color)
        aspect_score = sum(aspects.values()) / len(aspects)
        
        # Creamos barras horizontales con gradiente de color
        for i, (aspect, detected) in enumerate(aspects.items()):
            color = cmap(aspect_score * 0.7)  # Ajustamos el gradiente
            bar = ax.barh(aspect, [1], color=color if detected else "#F5F5F5")
            
            # Añadimos iconos y texto descriptivo
            if detected:
                ax.text(0.5, i, "✓", 
                       va='center', ha='center', 
                       color='white', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, i, "✗", 
                       va='center', ha='center', 
                       color='#9E9E9E', fontsize=12)
        
        # Personalización del gráfico
        ax.set_title('ANÁLISIS DE ASPECTOS', pad=20, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.close(fig)
        
        # 4. Resultado estructurado
        json_result = {
            "Resumen": {
                "Estrellas": result['label'],
                "Sentimiento": sentiment,
                "Confianza": f"{result['score']:.2%}",
                "Puntuación_Aspectos": f"{aspect_score:.0%}"
            },
            "Detalles": {
                "Texto_analizado": text[:200] + "..." if len(text) > 200 else text,
                "Aspectos": aspects
            }
        }
        
        return json_result, fig
        
    except Exception as e:
        return {"Error": f"Error en el análisis: {str(e)}"}, None

# Interfaz Gradio mejorada
iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="📝 Escribe tu reseña de producto", 
        placeholder="Ej: El producto es excelente pero no incluía los cables necesarios...",
        lines=3
    ),
    outputs=[
        gr.JSON(label="📊 Resultado del Análisis"),
        gr.Plot(label="📌 Aspectos Clave")
    ],
    examples=[
        ["Excelente producto. Funciona perfectamente en todos los juegos. Relación calidad-precio increíble!"],
        ["No cumple con lo esperado. No traía los cables de conexión y se calienta mucho."],
        ["Buen rendimiento pero tuve que comprar los cables por separado. En general está bien."]
    ],
    title="🛍️ Analizador Avanzado de Reseñas",
    description="""Analiza sentimientos y detecta aspectos clave en reseñas de productos en español.
    Detecta: Rendimiento, Calidad-Precio, Completitud y Recomendación""",
    allow_flagging="never"
)

# Configuración del lanzamiento
iface.launch(
    server_port=7860,
    show_error=True,
    share=False  # Cambia a True si quieres un enlace público temporal
)