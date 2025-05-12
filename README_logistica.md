
# 🚢 Regresión Logística – Dataset del Titanic

## 🎯 Objetivo del Modelo

El objetivo de este proyecto es predecir si un pasajero del Titanic **sobrevivió (`Survived = 1`) o no (`Survived = 0`)** utilizando un modelo de **regresión logística**, basado en información demográfica y de viaje.

---

## ⚙️ Variables Utilizadas

Las variables predictoras utilizadas fueron:

- `Pclass`: Clase del pasajero (1 = alta, 2 = media, 3 = baja)
- `Sex`: Sexo del pasajero (codificado)
- `Age`: Edad
- `SibSp`: Nº de hermanos/esposas a bordo
- `Parch`: Nº de padres/hijos a bordo
- `Fare`: Tarifa pagada
- `Embarked`: Puerto de embarque (codificado)

Variable objetivo:
- `Survived`: 0 = no sobrevivió, 1 = sobrevivió

---

## 🔧 Preparación del Modelo

- Se eliminaron columnas no relevantes (`Name`, `Ticket`, `Cabin`, `PassengerId`)
- Se rellenaron valores nulos en `Age` y `Embarked`
- Codificación de variables categóricas (`Sex`, `Embarked`) con `LabelEncoder`
- División del dataset: 80% entrenamiento / 20% validación

---

## 📈 Resultados del Modelo

- **Accuracy**: `0.8101`

### 🧮 Matriz de Confusión

|               | Predicho 0 | Predicho 1 |
|---------------|------------|------------|
| Real 0 (no)   | 90         | 15         |
| Real 1 (sí)   | 19         | 55         |

---

### 📋 Reporte de Clasificación

| Clase | Precision | Recall | F1-score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.83      | 0.86   | 0.84     | 105     |
| 1     | 0.79      | 0.74   | 0.76     | 74      |
| Macro avg | 0.81  | 0.80   | 0.80     | 179     |
| Weighted avg | 0.81 | 0.81 | 0.81     | 179     |

---

## 📌 Explicación de Métricas

- **Precision**: Proporción de predicciones positivas correctas.
- **Recall**: Capacidad del modelo de encontrar todos los positivos reales.
- **F1-score**: Media armónica entre precisión y recall.
- **Support**: Número real de instancias por clase.

---

## 🧠 Interpretación

- El modelo predice mejor los casos de **no supervivencia (clase 0)**.
- Tiene un buen balance en precisión y recall, con **F1-score de 0.76** para sobrevivientes.
- Buen desempeño como modelo base, con potencial de mejora.

---

## 🚀 Posibles Mejoras

- Añadir ingeniería de variables (ej. familia total, extracción de títulos del nombre)
- Probar modelos más complejos (árboles de decisión, XGBoost)
- Aplicar validación cruzada
- Normalización de variables como `Fare` y `Age`

---

## 🧾 Librerías utilizadas

```python
pandas
scikit-learn
```

---

## 📁 Fuente de datos

El archivo `train.csv` es parte del dataset de Titanic disponible públicamente en [Kaggle](https://www.kaggle.com/competitions/titanic).
