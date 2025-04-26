import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from fake_useragent import UserAgent
from transformers import pipeline
from tqdm import tqdm
import os

# Configuración
os.environ["OMP_NUM_THREADS"] = "1"
ua = UserAgent()

# 1. Scraper mejorado para Mercado Libre 2024
def scrape_mercado_libre(url, max_opiniones=20):
    headers = {'User-Agent': ua.random}
    opiniones = []
    
    try:
        # Primera página
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer de la nueva estructura
        reviews = soup.select('.ui-review-capability-comments__comment, .review')
        
        for review in reviews:
            try:
                texto = review.select_one('.ui-review-capability-comments__comment__content, .review-content').text.strip()
                estrellas = len(review.select('.ui-review-capability-ratings__star--on, .review-star-on'))
                opiniones.append({'texto': texto, 'estrellas': estrellas})
            except:
                continue
            
            if len(opiniones) >= max_opiniones:
                break
            
        time.sleep(random.uniform(1, 3))  # Evitar bloqueos
        
    except Exception as e:
        print(f"Error en scraping: {str(e)}")
    
    return opiniones

# 2. Cargar modelo de análisis
try:
    model = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1,
        truncation=True
    )
except Exception as e:
    print(f"Error cargando modelo: {str(e)}")
    exit()

# 3. Función de análisis optimizada
def analizar_opinion(texto):
    try:
        result = model(texto[:512])[0]
        stars = int(result['label'][0])
        return "POSITIVO" if stars >= 4 else "NEUTRO" if stars == 3 else "NEGATIVO"
    except:
        return "ERROR"

# 4. Procesamiento completo con manejo de errores
def analizar_producto(url):
    print("\n🔍 Extrayendo opiniones (puede tomar unos segundos)...")
    
    opiniones = scrape_mercado_libre(url)
    if not opiniones:
        print("\n❌ No se encontraron opiniones. Posibles causas:")
        print("- El producto no tiene opiniones públicas")
        print("- Mercado Libre ha cambiado su estructura HTML")
        print("- Bloqueo temporal por scraping (espera 10 minutos)")
        print("\n💡 Solución alternativa: Exporta opiniones manualmente a CSV y usa:")
        print("python analizar_csv.py opiniones.csv")
        return
    
    print(f"📊 Analizando {len(opiniones)} opiniones...")
    for opinion in tqdm(opiniones, desc="Progreso"):
        opinion['sentimiento'] = analizar_opinion(opinion['texto'])
        time.sleep(0.5)  # Evitar sobrecarga del modelo
    
    # Crear DataFrame
    df = pd.DataFrame(opiniones)
    
    # 5. Visualización mejorada
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df['sentimiento'].value_counts()
    colors = {'POSITIVO': '#2ecc71', 'NEGATIVO': '#e74c3c', 'NEUTRO': '#f39c12', 'ERROR': '#95a5a6'}
    
    bars = counts.plot(
        kind='bar', 
        color=[colors[x] for x in counts.index],
        edgecolor='black',
        ax=ax
    )
    
    # Añadir valores encima de las barras
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())}",
            ha='center',
            va='bottom'
        )
    
    ax.set_title('ANÁLISIS DE OPINIONES - MERCADO LIBRE', pad=20, fontweight='bold')
    ax.set_xlabel('Sentimiento')
    ax.set_ylabel('Cantidad')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 6. Mostrar resultados
    print("\n📌 RESUMEN ESTADÍSTICO:")
    print(f"Total opiniones analizadas: {len(df)}")
    print(f"✅ Positivas: {counts.get('POSITIVO', 0)} ({counts.get('POSITIVO', 0)/len(df):.1%})")
    print(f"⚠️ Neutras: {counts.get('NEUTRO', 0)}")
    print(f"❌ Negativas: {counts.get('NEGATIVO', 0)}")
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f'resultados_opiniones_{timestamp}.csv', index=False)
    plt.savefig(f'analisis_sentimientos_{timestamp}.png', dpi=300)
    
    print("\n💾 Resultados guardados en:")
    print(f"- resultados_opiniones_{timestamp}.csv")
    print(f"- analisis_sentimientos_{timestamp}.png")
    
    plt.show()

# Ejecución
if __name__ == "__main__":
    print("🛒 ANALIZADOR DE OPINIONES - MERCADO LIBRE")
    print("----------------------------------------")
    
    # URL de ejemplo (reemplazar por la real)
    url = input("Ingresa la URL completa del producto en Mercado Libre: ").strip()
    
    if not url.startswith('https://www.mercadolibre.com'):
        print("\n⚠️ Error: URL debe ser de Mercado Libre (ej: https://www.mercadolibre.com.mx/...)")
    else:
        analizar_producto(url)