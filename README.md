# Predicción de Enfermedades mediante Machine Learning

Este proyecto tiene como objetivo predecir si un paciente tiene una enfermedad específica utilizando dos modelos de machine learning: Árbol de Decisión y Redes Neuronales. Para ello, se han utilizado datos médicos como análisis de corazón, sangre, niveles de azúcar, entre otros.

## Descripción del Proyecto

Este proyecto utiliza un conjunto de datos de pacientes que incluye diversas características clínicas como análisis de corazón, sangre y niveles de azúcar para predecir si un paciente padece una enfermedad. El análisis se realiza utilizando dos modelos de machine learning: Árbol de Decisión y Redes Neuronales. Ambos modelos son comparados para determinar cuál ofrece mejores resultados en la predicción.

## Modelos
### Árbol de Decisión
El modelo de Árbol de Decisión se utiliza para generar reglas interpretables sobre los datos. Este modelo es entrenado utilizando scikit-learn.

Ventajas: Fácil interpretación y relativamente rápido en el entrenamiento.
Desventajas: Puede sobreajustarse en los datos, especialmente si hay muchas características irrelevantes.
### Redes Neuronales
El modelo de Redes Neuronales es implementado utilizando la librería TensorFlow. Este modelo es ideal para capturar patrones no lineales en los datos, pero puede requerir más tiempo de entrenamiento y recursos computacionales.

Ventajas: Capaz de capturar relaciones complejas en los datos.
Desventajas: Mayor tiempo de entrenamiento y difícil interpretación.
