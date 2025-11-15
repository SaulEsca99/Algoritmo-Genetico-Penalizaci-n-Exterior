# Algoritmo Genetico con Penalización Exterior - Tarea 4

**Autor:** Escamilla Lazcano Saúl

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/Librerías-Matplotlib%20%7C%20NumPy-green.svg)](https://pypi.org/project/numpy/)

## 🎯 Descripción del Proyecto

Este proyecto implementa un **Algoritmo Genético (AG)** para resolver un problema de **optimización con restricciones**. El objetivo es minimizar una función sujeta a dos desigualdades (restricciones `g(x)`).

La característica principal de este script es el manejo de restricciones mediante el método de **Penalización Exterior**, donde el factor de penalización (`λ_P`) se ajusta de forma **adaptativa** para guiar a la población hacia la región factible.

Además, el script incluye un **dashboard de visualización en tiempo real** construido con `Matplotlib` que muestra la evolución de la población y las métricas de rendimiento generación por generación.

## 📈 El Problema de Optimización

El objetivo es encontrar el mínimo de la siguiente función:

**Función Objetivo (Minimizar):**
$$ f(x,y) = 4(x-3)^2 + 3(y-3)^2 $$

**Sujeto a las restricciones:**
$$ g_1(x,y) = 2x + y - 2 \le 0 $$
$$ g_2(x,y) = 3x + 4y - 6 \le 0 $$

**Límites del Espacio de Búsqueda:**
$$ x \in [0, 1] $$
$$ y \in [0, 2] $$

## ⚙️ Método: Penalización Exterior Adaptativa

Para manejar las restricciones, la función de aptitud se transforma en una **función penalizada** `F_P(x,y)`. El algoritmo no minimiza `f(x)` directamente, sino `F_P(x,y)`:

**Función Penalizada:**
$$ F_P(x) = f(x) + \lambda_P \cdot P(x) $$

Donde `P(x)` es la penalización, que solo se activa si una restricción es violada:

**Término de Penalización:**
$$ P(x) = \sum_{i=1}^{2} (\max(0, g_i(x)))^2 $$

### Ajuste Adaptativo
El script incluye una variable `ajuste_adaptativo = True`. Si después de 10 generaciones no se encuentra ninguna solución factible, el factor de penalización `λ_P` se duplica, presionando más fuerte a la población para que respete las restricciones.

## 🧬 Arquitectura del Algoritmo Genético

* **Codificación:** Binaria (cálculo de bits por precisión).
* **Selección:** Torneo Determinista.
* **Cruzamiento:** Dos Puntos (con probabilidad `pc`).
* **Mutación:** Simple (Bit Flip, con probabilidad `pm`).
* **Sustitución:** **Por Familia** (De una familia de 2 padres y 2 hijos, solo los 2 mejores (con menor `F_P`) pasan a la siguiente generación).

## 🚀 Cómo Ejecutar

1.  Asegúrate de tener las dependencias instaladas.
2.  Ejecuta el script desde tu terminal:
    ```bash
    python tu_script.py
    ```
    *(Reemplaza `tu_script.py` con el nombre de tu archivo)*

3.  ¡Observa la ventana de Matplotlib! La simulación comenzará automáticamente.

### Ajuste de Velocidad
Puedes controlar la velocidad de la animación cambiando la variable `VELOCIDAD` al final del script:

```python
# ============================================================================
# CONFIGURACIÓN DE VELOCIDAD DE ANIMACIÓN
# ============================================================================
VELOCIDAD = 1  # <-- CAMBIA ESTE VALOR
#   0.1 = Muy rápido
#   0.5 = Normal (recomendado)
#   1.0 = Lento
# ============================================================================
```

## 📊 Visualización y Resultados

Al ejecutar el script, se abre un dashboard en vivo que muestra:

1.  **Espacio de Búsqueda:** Un gráfico 2D con las curvas de nivel de `f(x)`, las líneas de restricción, la **región factible** (verde claro), y la población (verde para factibles, rojo para no factibles).
2.  **Evolución del Fitness:** El valor `F_P` (penalizado) del mejor individuo de cada generación.
3.  **f(x) y P(x):** Los valores separados de la función objetivo (`f`) y la penalización (`P`) del mejor individuo.
4.  **% Factibles:** El porcentaje de la población que se encuentra dentro de la región factible.
5.  **Panel de Información:** Un resumen de texto con los valores del mejor individuo de la generación actual.

---
*(Te recomiendo ejecutar el script, tomar un screenshot del dashboard y reemplazar esta línea y la siguiente por esa imagen)*

**[Screenshot del dashboard en Matplotlib]**
---

### Salida Final en Consola
Una vez que el AG termina o cierras la ventana, el script imprime un resumen detallado de la mejor solución encontrada:

```
======================================================================
RESULTADOS FINALES
======================================================================
λ_P final utilizado: 1000

Mejor solución encontrada:
  x = 0.000000
  y = 1.500000

Valores:
  f(x,y) = 42.750000
  P(x,y) = 0.000000
  F_P(x,y) = 42.750000

Restricciones:
  g1(x,y) = -0.500000 ✓
  g2(x,y) = 0.000000 ✓

Estado: ✓ SOLUCIÓN FACTIBLE
======================================================================
```

## 📋 Dependencias

* **Python 3.x**
* **Matplotlib**
* **NumPy**

Puedes instalarlas usando `pip`:
```bash
pip install matplotlib numpy
```
