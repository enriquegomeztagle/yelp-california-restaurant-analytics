# Yelp California Restaurant Analytics

Dashboard interactivo de análisis exploratorio de datos (EDA) y predicción de popularidad para restaurantes en California usando datos de Yelp.

## 🎯 Características

- **Visualizaciones interactivas** de distribución de calificaciones, reseñas y categorías
- **Análisis por ciudad** de promedios y proporciones de alta calificación
- **Modelo de Machine Learning** para predecir la popularidad de restaurantes
- **Dashboard intuitivo** construido con Streamlit

## 📊 Visualizaciones

1. **Distribución de calificaciones** - Representación visual con estrellas proporcionales
2. **Distribución de reseñas** - Histograma del número de reseñas
3. **Relación reseñas vs calificaciones** - Scatter plot
4. **Top 10 categorías** - Gráfico de barras
5. **Promedio por ciudad** - Barras horizontales
6. **Proporción de altas calificaciones** - Gráfico circular

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/enriquegomeztagle/yelp-california-restaurant-analytics.git
cd yelp-california-restaurant-analytics

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

```bash
streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🤖 Modelo de Predicción

El modelo utiliza **Gradient Boosting Classifier** para predecir si un restaurante será popular (número de reseñas por encima de la mediana).

**Features utilizadas:**
- Calificación (estrellas)
- Número de categorías
- Ubicación (latitud, longitud)
- Ciudad
- Estado (abierto/cerrado)
- Categorías específicas del restaurante

## 📦 Dependencias

- streamlit
- pandas
- plotly
- scikit-learn

## 📁 Estructura del Proyecto

```
yelp-california-restaurant-analytics/
├── streamlit_app.py          # Aplicación principal
├── filtered_yelp_CA.csv      # Dataset de restaurantes
├── requirements.txt          # Dependencias
├── .gitignore               # Archivos ignorados
└── README.md                # Documentación
```

## 📈 Dataset

El dataset contiene información de restaurantes en California extraída de Yelp, incluyendo:
- Calificaciones (stars)
- Número de reseñas
- Ubicación geográfica
- Categorías de restaurante
- Estado operacional

## 🛠️ Tecnologías

- **Python 3.x**
- **Streamlit** - Framework para la aplicación web
- **Plotly** - Visualizaciones interactivas
- **Scikit-learn** - Modelo de Machine Learning
- **Pandas** - Manipulación de datos

## 👤 Autor

Enrique Gómez Tagle

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
