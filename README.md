# 🚢 Predicción de Supervivencia del Titanic

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.79.1-009688?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.0-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/Licencia-MIT-green)

**Aplicación de Machine Learning para predecir la supervivencia de pasajeros del Titanic**

*Trabajo Final - Cloud Computing 2026*

</div>

---

## 👥 Equipo de Desarrollo

| Integrante | Rol |
|------------|-----|
| **Alejandra Véliz** | Desarrollador |
| **Juan Pablo Lucero** | Desarrollador |
| **Leonor Saravia** | Desarrollador |

---

## 📋 Descripción del Proyecto

Este proyecto implementa una **API REST** basada en FastAPI que utiliza modelos de Machine Learning para predecir si un pasajero del Titanic habría sobrevivido al naufragio, basándose en características como clase del pasajero, género, edad, y otros factores históricos.

### 🎯 Objetivo

Desarrollar un servicio web desplegable en la nube que permita realizar predicciones de supervivencia utilizando dos modelos de clasificación:

- **Regresión Logística**
- **Bosque Aleatorio (Random Forest)**

---

## 🏗️ Arquitectura del Proyecto

```
CC_2026_TrabajoFinal/
│
├── app.py                         # Aplicación FastAPI principal
├── modelo_regresion_logistica.pkl # Modelo de regresión logística entrenado
├── modelo_bosque_aleatorio.pkl    # Modelo de bosque aleatorio entrenado
├── scaler.pkl                     # Escalador para normalización de datos
├── requirements.txt               # Dependencias del proyecto
├── Procfile                       # Configuración para despliegue (Heroku/Railway)
├── runtime.txt                    # Versión de Python requerida
├── LICENSE                        # Licencia MIT
└── README.md                      # Este archivo
```

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/LeoSaraviaS/CC_2026_TrabajoFinal.git
cd CC_2026_TrabajoFinal
```

### Paso 2: Crear un Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la Aplicación

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

La API estará disponible en: `http://localhost:5000`

---

## 📖 Uso de la API

### Documentación Interactiva

Una vez que la aplicación esté corriendo, puedes acceder a la documentación interactiva de Swagger UI:

- **Swagger UI:** `http://localhost:5000/docs`
- **ReDoc:** `http://localhost:5000/redoc`

### Endpoint de Predicción

#### `POST /predict/`

Realiza una predicción de supervivencia para un pasajero.

**Request Body:**

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 211.3375,
  "Embarked": "S"
}
```

**Parámetros:**

| Campo | Tipo | Descripción | Valores Válidos |
|-------|------|-------------|-----------------|
| `Pclass` | int | Clase del pasajero | 1 (1ra clase), 2 (2da clase), 3 (3ra clase) |
| `Sex` | string | Género del pasajero | `male`, `female`, `hombre`, `mujer` |
| `Age` | float | Edad del pasajero | Número decimal (ej: 22.5) |
| `SibSp` | int | Número de hermanos/cónyuges a bordo | 0, 1, 2, ... |
| `Parch` | int | Número de padres/hijos a bordo | 0, 1, 2, ... |
| `Fare` | float | Tarifa pagada por el billete | Número decimal (ej: 50.0) |
| `Embarked` | string | Puerto de embarque | `S` (Southampton), `C` (Cherbourg), `Q` (Queenstown) |

**Response:**

```json
{
  "Sobrevive": true,
  "ProbabilidadSupervivencia": 0.89,
  "Mensaje": "El pasajero SOBREVIVE"
}
```

### Ejemplo con cURL

```bash
curl -X POST "http://localhost:5000/predict/" \
     -H "Content-Type: application/json" \
     -d '{
           "Pclass": 1,
           "Sex": "female",
           "Age": 29,
           "SibSp": 0,
           "Parch": 0,
           "Fare": 211.34,
           "Embarked": "S"
         }'
```

### Ejemplo con Python

```python
import requests

url = "http://localhost:5000/predict/"
pasajero = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(url, json=pasajero)
print(response.json())
```

---

## ☁️ Despliegue en la Nube

### Heroku

El proyecto incluye los archivos necesarios para despliegue en Heroku:

1. **Procfile:** Define el comando de inicio
2. **runtime.txt:** Especifica la versión de Python
3. **requirements.txt:** Lista de dependencias

```bash
# Instalar Heroku CLI y autenticarse
heroku login

# Crear una nueva aplicación
heroku create nombre-de-tu-app

# Desplegar
git push heroku main

# Abrir la aplicación
heroku open
```

### Railway / Render

También es compatible con otras plataformas de despliegue como Railway o Render que detectan automáticamente la configuración de Python.

---

## 🔧 Tecnologías Utilizadas

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **Python** | 3.10 | Lenguaje de programación |
| **FastAPI** | 0.79.1 | Framework web para la API |
| **Uvicorn** | 0.18.2 | Servidor ASGI |
| **Scikit-Learn** | 1.7.0 | Modelos de Machine Learning |
| **Pandas** | 2.3.0 | Manipulación de datos |
| **NumPy** | 2.2.6 | Cálculos numéricos |
| **Pydantic** | 1.10.22 | Validación de datos |

---

## 📊 Modelos de Machine Learning

### Regresión Logística
- Modelo lineal para clasificación binaria
- Ideal para entender la importancia de cada variable
- Rápido y eficiente

### Bosque Aleatorio (Random Forest)
- Ensemble de árboles de decisión
- Mayor capacidad de capturar relaciones no lineales
- Más robusto ante outliers

---

## 📁 Variables del Dataset

El modelo fue entrenado con las siguientes características del dataset original del Titanic:

| Variable | Descripción |
|----------|-------------|
| **Pclass** | Clase socioeconómica (1 = Alta, 2 = Media, 3 = Baja) |
| **Sex** | Género del pasajero |
| **Age** | Edad en años |
| **SibSp** | Número de hermanos/cónyuges a bordo |
| **Parch** | Número de padres/hijos a bordo |
| **Fare** | Tarifa del pasaje en libras |
| **Embarked** | Puerto de embarque |

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

---

## 📞 Contacto

Para consultas sobre el proyecto, puedes contactar a cualquier miembro del equipo a través del repositorio de GitHub.

---

<div align="center">

**Universidad Abierta Interamericana (UAI)**

*Cloud Computing - Trabajo Final 2026*

⭐ ¡Si te gustó el proyecto, no olvides dejar una estrella! ⭐

</div>
