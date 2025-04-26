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
from datetime import datetime

# Configuración mejorada
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ua = UserAgent()

# 1. Scraper optimizado con manejo de errores mejorado
def scrape_mercado_libre(url, max_opiniones=20):
    headers = {'User-Agent': ua.random}
    opiniones = []
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Selectores actualizados para 2024
        reviews = soup.select('.ui-review-capability-comments__comment, .review, .ui-pdp-review__content')
        
        for review in reviews[:max_opiniones]:
            try:
                texto = review.select_one(
                    '.ui-review-capability-comments__comment__content, '
                    '.review-content, '
                    '.ui-pdp-review__content__comment'
                ).text.strip()
                
                # Extraer estrellas de diferentes estructuras
                estrellas = len(review.select('.ui-review-capability-ratings__star--on, .review-star-on, .ui-pdp-review__rating__star--on'))
                estrellas = estrellas if estrellas > 0 else None
                
                opiniones.append({
                    'texto': texto,
                    'estrellas': estrellas,
                    'fecha_analisis': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                continue
            
        time.sleep(random.uniform(1, 3))
        
    except requests.RequestException as e:
        print(f"\n⚠️ Error al conectarse a Mercado Libre: {str(e)}")
    except Exception as e:
        print(f"\n⚠️ Error inesperado: {str(e)}")
    
    return opiniones

# 2. Carga del modelo con caché
try:
    model = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1,
        truncation=True
    )
except Exception as e:
    print(f"\n❌ Error cargando el modelo de IA: {str(e)}")
    exit()

# 3. Análisis de sentimiento con puntuación
def analizar_opinion(texto):
    try:
        result = model(texto[:512])[0]
        stars = int(result['label'][0])
        return {
            'sentimiento': "POSITIVO" if stars >= 4 else "NEUTRO" if stars == 3 else "NEGATIVO",
            'confianza': f"{result['score']:.0%}",
            'estrellas': stars
        }
    except Exception as e:
        return {'sentimiento': "ERROR", 'confianza': "0%", 'estrellas': None}

# 4. Visualización mejorada con Plotly (interactiva)
def generar_visualizacion(df):
    try:
        import plotly.express as px
        
        # Gráfico interactivo
        fig = px.pie(
            df, 
            names='sentimiento', 
            title='Distribución de Sentimientos',
            color='sentimiento',
            color_discrete_map={
                'POSITIVO': '#2ecc71',
                'NEGATIVO': '#e74c3c',
                'NEUTRO': '#f39c12',
                'ERROR': '#95a5a6'
            },
            hole=0.3
        )
        
        # Guardar como HTML interactivo
        fig.write_html("analisis_interactivo.html")
        print("\n📊 Gráfico interactivo guardado como 'analisis_interactivo.html'")
        
    except ImportError:
        # Fallback a matplotlib si Plotly no está disponible
        plt.style.use('ggplot')
        counts = df['sentimiento'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6'][:len(counts)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
        
        ax.set_title('Distribución de Sentimientos', pad=20, fontweight='bold')
        ax.set_xlabel('Sentimiento')
        ax.set_ylabel('Cantidad')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analisis_sentimientos.png', dpi=120)
        print("\n📈 Gráfico estático guardado como 'analisis_sentimientos.png'")

# 5. Procesamiento completo
def analizar_producto(url):
    print("\n🔍 Extrayendo opiniones (puede tomar unos segundos)...")
    
    opiniones = scrape_mercado_libre(url)
    if not opiniones:
        print("\n❌ No se encontraron opiniones. Prueba:")
        print("- Verificar que la URL sea correcta")
        print("- Intentar manualmente con 'python analizar_csv.py tus_opiniones.csv'")
        return
    
    print(f"\n📊 Analizando {len(opiniones)} opiniones...")
    resultados = []
    for opinion in tqdm(opiniones, desc="Progreso"):
        analysis = analizar_opinion(opinion['texto'])
        resultados.append({**opinion, **analysis})
        time.sleep(0.5)  # Evitar saturación
    
    df = pd.DataFrame(resultados)
    
    # Estadísticas
    print("\n📌 RESUMEN ESTADÍSTICO:")
    print(f"Total opiniones analizadas: {len(df)}")
    positivas = df[df['sentimiento'] == 'POSITIVO']
    print(f"✅ Positivas: {len(positivas)} ({len(positivas)/len(df):.1%})")
    print(f"⚠️ Neutras: {len(df[df['sentimiento'] == 'NEUTRO'])}")
    print(f"❌ Negativas: {len(df[df['sentimiento'] == 'NEGATIVO'])}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"resultados_opiniones_{timestamp}"
    
    df.to_csv(f"{nombre_archivo}.csv", index=False, encoding='utf-8-sig')
    print(f"\n💾 Resultados guardados en:")
    print(f"- {nombre_archivo}.csv (datos completos)")
    
    generar_visualizacion(df)
    
    # Mostrar ejemplo de análisis
    print("\n🔎 Ejemplo de análisis realizado:")
    print(f"Texto: {df.iloc[0]['texto'][:100]}...")
    print(f"Sentimiento: {df.iloc[0]['sentimiento']} ({df.iloc[0]['confianza']} de confianza)")

# Ejecución principal
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🛒 ANALIZADOR AVANZADO DE OPINIONES - MERCADO LIBRE")
    print("="*50)
    
    # Verificar dependencias
    try:
        import plotly
        print("\nℹ️ Se usará Plotly para gráficos interactivos (mejor experiencia)")
    except ImportError:
        print("\nℹ️ Plotly no está instalado. Usando matplotlib para gráficos estáticos")
        print("   Para gráficos interactivos: pip install plotly")
    
    url = input("\n📌 Ingresa la URL completa del producto en Mercado Libre: ").strip()
    
    if not url.startswith(('https://www.mercadolibre.com', 'http://www.mercadolibre.com')):
        print("\n⚠️ Error: La URL debe ser de Mercado Libre (ej: https://www.mercadolibre.com.ar/...)")
    else:
        analizar_producto(url)
    
    print("\n🎯 Análisis completado. Puedes mejorar el programa con:")
    print("- pip install plotly (para gráficos interactivos)")
    print("- Más ejemplos en: https://github.com/ejemplos-analisis-sentimientos")