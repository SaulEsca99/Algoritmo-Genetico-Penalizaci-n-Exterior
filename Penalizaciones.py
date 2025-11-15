# Escamilla Lazcano Saúl - Grupo 5BV1
# Tarea 4: Algoritmo Genético con Penalización Exterior
# Problema: min f(x,y) = 4(x-3)² + 3(y-3)² sujeto a restricciones

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 1. Cálculo de longitud de bits
def calcular_longitud_bits(lim_inf, lim_sup, precision):
    """L = int[log2((ls - li) * 10^precisión) + 0.9]"""
    rango = lim_sup - lim_inf
    L = int(math.log2(rango * (10 ** precision)) + 0.9)
    return L


# 2. Generación de población inicial binaria
def generar_poblacion_inicial(tam_poblacion, limites, precision):
    """Genera población binaria inicial"""
    longitudes_bits = []
    for lim_inf, lim_sup in limites:
        L = calcular_longitud_bits(lim_inf, lim_sup, precision)
        longitudes_bits.append(L)
    longitud_total = sum(longitudes_bits)
    poblacion = []
    for _ in range(tam_poblacion):
        individuo = ''.join(str(random.randint(0, 1)) for _ in range(longitud_total))
        poblacion.append(individuo)
    return poblacion, longitudes_bits


# 3. Decodificación binario a reales
def decodificar_individuo(individuo, longitudes_bits, limites):
    """x_real = li + [x_entero × (ls - li)] / (2^L - 1)"""
    valores_reales = []
    inicio = 0
    for i, (lim_inf, lim_sup) in enumerate(limites):
        fin = inicio + longitudes_bits[i]
        segmento_binario = individuo[inicio:fin]
        valor_entero = int(segmento_binario, 2)
        x_real = lim_inf + (valor_entero * (lim_sup - lim_inf)) / (2 ** longitudes_bits[i] - 1)
        valores_reales.append(x_real)
        inicio = fin
    return valores_reales


# 4. Selección por torneo
def seleccion_torneo(poblacion, aptitudes, tam_poblacion):
    """Implementa selección por torneo determinista"""
    padres = []
    permutacion1 = list(range(tam_poblacion))
    permutacion2 = list(range(tam_poblacion))
    random.shuffle(permutacion1)
    random.shuffle(permutacion2)

    for i in range(tam_poblacion):
        competidor_izq = permutacion1[i]
        competidor_der = permutacion2[i]

        if aptitudes[competidor_izq] <= aptitudes[competidor_der]:
            padres.append(poblacion[competidor_izq])
        else:
            padres.append(poblacion[competidor_der])

    return padres


# 5. Cruzamiento en dos puntos
def cruzamiento_dos_puntos(padres, pc):
    """Implementa cruzamiento en dos puntos"""
    hijos = []
    tam_poblacion = len(padres)

    for i in range(0, tam_poblacion - 1, 2):
        padre1 = padres[i]
        padre2 = padres[i + 1]

        if random.random() <= pc:
            longitud = len(padre1)
            punto1 = random.randint(1, longitud - 2)
            punto2 = random.randint(punto1 + 1, longitud - 1)

            hijo1 = padre1[:punto1] + padre2[punto1:punto2] + padre1[punto2:]
            hijo2 = padre2[:punto1] + padre1[punto1:punto2] + padre2[punto2:]
        else:
            hijo1 = padre1
            hijo2 = padre2

        hijos.extend([hijo1, hijo2])

    return hijos


# 6. Mutación simple
def mutacion_simple(poblacion, pm):
    """Implementa mutación simple (bit flip)"""
    poblacion_mutada = []

    for individuo in poblacion:
        rand = random.random()

        if rand <= pm:
            pto = random.randint(0, len(individuo) - 1)
            individuo_mutado = list(individuo)
            individuo_mutado[pto] = '1' if individuo_mutado[pto] == '0' else '0'
            poblacion_mutada.append(''.join(individuo_mutado))
        else:
            poblacion_mutada.append(individuo)

    return poblacion_mutada


# 7. Sustitución por familia
def sustitucion_por_familia(padres, aptitudes_padres, hijos, aptitudes_hijos):
    """
    Sustitución por familia según pseudocódigo del PDF:
    De cada familia (2 padres + 2 hijos), pasan los 2 mejores
    """
    nueva_poblacion = []
    nuevas_aptitudes = []
    tam_poblacion = len(padres)

    for i in range(0, tam_poblacion, 2):
        # Familia = [Padre1, Padre2, Hijo1, Hijo2]
        familia = [
            padres[i], padres[i + 1],
            hijos[i], hijos[i + 1]
        ]

        aptitudes_familia = [
            aptitudes_padres[i], aptitudes_padres[i + 1],
            aptitudes_hijos[i], aptitudes_hijos[i + 1]
        ]

        # Ordenar familia por aptitud (mejor a peor)
        familia_con_aptitudes = list(zip(familia, aptitudes_familia))
        familia_con_aptitudes.sort(key=lambda x: x[1])

        # Los 2 mejores pasan a la nueva población
        mejor1, aptitud1 = familia_con_aptitudes[0]
        mejor2, aptitud2 = familia_con_aptitudes[1]

        nueva_poblacion.extend([mejor1, mejor2])
        nuevas_aptitudes.extend([aptitud1, aptitud2])

    return nueva_poblacion, nuevas_aptitudes

# 8. Función objetivo del problema
def funcion_objetivo(x, y):
    """f(x,y) = 4(x-3)² + 3(y-3)²"""
    return 4 * (x - 3) ** 2 + 3 * (y - 3) ** 2


# 9. Restricciones del problema
def restriccion_g1(x, y):
    """g1(x,y) = 2x + y - 2 ≤ 0"""
    return 2 * x + y - 2


def restriccion_g2(x, y):
    """g2(x,y) = 3x + 4y - 6 ≤ 0"""
    return 3 * x + 4 * y - 6


# 10. Función de penalización exterior
def calcular_penalizacion(x, y):
    """
    P(x) = Σ max(gi(x), 0)² + Σ hj(x)²
    Solo tenemos restricciones de desigualdad (g1 y g2)
    """
    g1 = restriccion_g1(x, y)
    g2 = restriccion_g2(x, y)

    penalizacion = max(g1, 0) ** 2 + max(g2, 0) ** 2
    return penalizacion


# 11. Función penalizada
def funcion_penalizada(x, y, lambda_p):
    """F_P(x) = f(x) + λ_P * P(x)"""
    f_val = funcion_objetivo(x, y)
    p_val = calcular_penalizacion(x, y)
    fp_val = f_val + lambda_p * p_val
    return fp_val


# 12. Verificar si individuo es factible
def es_factible(x, y):
    """Verifica si satisface todas las restricciones"""
    return restriccion_g1(x, y) <= 1e-6 and restriccion_g2(x, y) <= 1e-6


# 13. Evaluar población con penalización
def evaluar_poblacion_penalizada(poblacion, longitudes_bits, limites, lambda_p):
    """Evalúa población usando función penalizada"""
    resultados = []
    for individuo in poblacion:
        valores_reales = decodificar_individuo(individuo, longitudes_bits, limites)
        x, y = valores_reales

        # Calcular todos los valores
        f_val = funcion_objetivo(x, y)
        p_val = calcular_penalizacion(x, y)
        fp_val = funcion_penalizada(x, y, lambda_p)
        factible = es_factible(x, y)

        resultados.append({
            'individuo': individuo,
            'x': x,
            'y': y,
            'f': f_val,
            'P': p_val,
            'Fp': fp_val,
            'factible': factible
        })

    return resultados


# 14. Encontrar mejor individuo
def encontrar_mejor_individuo(resultados):
    """Encuentra el individuo con mejor aptitud (menor Fp)"""
    return min(resultados, key=lambda x: x['Fp'])



# VISUALIZACIÓN EN TIEMPO REAL

def preparar_visualizacion():
    """Prepara la figura con todos los subplots - LAYOUT MEJORADO"""
    fig = plt.figure(figsize=(18, 11))

    # Grid mejorado: 4 filas, 4 columnas
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.4,
                          left=0.05, right=0.98, top=0.94, bottom=0.04,
                          height_ratios=[1, 1, 0.8, 0.5])

    # Espacio de búsqueda (más grande) - 2 filas x 2 columnas
    ax_espacio = fig.add_subplot(gs[0:2, 0:2])

    # Gráficas de evolución - lado derecho
    ax_fitness = fig.add_subplot(gs[0, 2:4])  # Arriba derecha
    ax_valores = fig.add_subplot(gs[1, 2:4])  # Centro derecha

    # Gráfica de factibles - abajo izquierda (solo 2 columnas)
    ax_factibles = fig.add_subplot(gs[2, 0:2])

    # Panel de información - abajo DERECHA (solo 2 columnas)
    ax_info = fig.add_subplot(gs[2:4, 2:4])  # Ocupa esquina inferior derecha
    ax_info.axis('off')

    return fig, ax_espacio, ax_fitness, ax_valores, ax_factibles, ax_info


def dibujar_espacio_busqueda(ax, limites):
    """Dibuja el espacio de búsqueda con restricciones"""
    x_vals = np.linspace(limites[0][0], limites[0][1], 300)
    y_vals = np.linspace(limites[1][0], limites[1][1], 300)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Función objetivo
    Z = 4 * (X - 3) ** 2 + 3 * (Y - 3) ** 2
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=7)

    # Restricciones
    G1 = 2 * X + Y - 2
    G2 = 3 * X + 4 * Y - 6

    # Dibujar líneas de restricciones (sin usar .collections deprecado)
    ax.contour(X, Y, G1, levels=[0], colors='red', linewidths=2, linestyles='--')
    ax.contour(X, Y, G2, levels=[0], colors='blue', linewidths=2, linestyles='--')

    # Agregar etiquetas manualmente con plot vacío
    ax.plot([], [], 'r--', linewidth=2, label='g₁=0')
    ax.plot([], [], 'b--', linewidth=2, label='g₂=0')

    # Región factible
    factible = (G1 <= 0) & (G2 <= 0)
    ax.contourf(X, Y, factible, levels=[0.5, 1.5], colors=['green'], alpha=0.15)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Espacio de Búsqueda con Restricciones', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(limites[0])
    ax.set_ylim(limites[1])


def actualizar_visualizacion(fig, axes, gen, resultados, historial, mejor_global, lambda_p):
    """Actualiza todos los gráficos"""
    ax_espacio, ax_fitness, ax_valores, ax_factibles, ax_info = axes

    # Limpiar axes dinámicos
    ax_espacio.clear()
    ax_fitness.clear()
    ax_valores.clear()
    ax_factibles.clear()
    ax_info.clear()
    ax_info.axis('off')

    # 1. Espacio de búsqueda
    limites = [(0, 1), (0, 2)]
    dibujar_espacio_busqueda(ax_espacio, limites)

    # Plotear población ACTUAL - Separar factibles y no factibles
    x_factibles = [res['x'] for res in resultados if res['factible']]
    y_factibles = [res['y'] for res in resultados if res['factible']]
    x_no_factibles = [res['x'] for res in resultados if not res['factible']]
    y_no_factibles = [res['y'] for res in resultados if not res['factible']]

    # Dibujar no factibles (rojos)
    if x_no_factibles:
        ax_espacio.scatter(x_no_factibles, y_no_factibles, c='red', marker='x',
                           s=100, alpha=0.7, linewidths=2, label='No factibles', zorder=5)

    # Dibujar factibles (verdes)
    if x_factibles:
        ax_espacio.scatter(x_factibles, y_factibles, c='lime', marker='o',
                           s=80, alpha=0.7, edgecolors='darkgreen', linewidths=1.5,
                           label='Factibles', zorder=6)

    # Mejor global (estrella dorada más grande)
    if mejor_global:
        ax_espacio.scatter(mejor_global['x'], mejor_global['y'], c='gold', marker='*',
                           s=500, edgecolors='black', linewidths=2.5, label='Mejor global', zorder=10)

    # Agregar leyenda
    ax_espacio.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # 2. Evolución del fitness
    ax_fitness.plot(historial['Fp'], 'b-', linewidth=2)
    ax_fitness.set_xlabel('Generación')
    ax_fitness.set_ylabel('F_P(x)')
    ax_fitness.set_title('Evolución del Fitness Penalizado')
    ax_fitness.grid(True, alpha=0.3)

    # 3. f(x) y P(x)
    ax_valores.plot(historial['f'], 'g-', linewidth=2, label='f(x)')
    ax_valores.plot(historial['P'], 'r-', linewidth=2, label='P(x)')
    ax_valores.set_xlabel('Generación')
    ax_valores.set_ylabel('Valor')
    ax_valores.set_title('f(x) y P(x) del Mejor')
    ax_valores.legend()
    ax_valores.grid(True, alpha=0.3)

    # 4. % Factibles
    ax_factibles.plot(historial['pct_factibles'], 'm-', linewidth=2)
    ax_factibles.set_xlabel('Generación')
    ax_factibles.set_ylabel('% Factibles')
    ax_factibles.set_title('Soluciones Factibles')
    ax_factibles.set_ylim([0, 105])
    ax_factibles.grid(True, alpha=0.3)
    ax_factibles.axhline(y=100, color='g', linestyle='--', alpha=0.5)

    # 5. Información
    mejor = encontrar_mejor_individuo(resultados)
    n_factibles = sum(1 for r in resultados if r['factible'])
    pct_fact = (n_factibles / len(resultados)) * 100

    info_text = f"""
╔═══════════════════════════════════════════════════════════╗
║  GENERACIÓN {gen + 1}
║  λ_P = {lambda_p}
╠═══════════════════════════════════════════════════════════╣
║  MEJOR DE LA GENERACIÓN:
║    • x = {mejor['x']:.6f},  y = {mejor['y']:.6f}
║    • f(x,y) = {mejor['f']:.6f}
║    • P(x,y) = {mejor['P']:.6f}
║    • F_P(x,y) = {mejor['Fp']:.6f}
║    • Estado: {'✓ FACTIBLE' if mejor['factible'] else '✗ NO FACTIBLE'}
║
║  RESTRICCIONES:
║    • g1 = {restriccion_g1(mejor['x'], mejor['y']):.6f} {'✓' if restriccion_g1(mejor['x'], mejor['y']) <= 0 else '✗'}
║    • g2 = {restriccion_g2(mejor['x'], mejor['y']):.6f} {'✓' if restriccion_g2(mejor['x'], mejor['y']) <= 0 else '✗'}
║
║  POBLACIÓN:
║    • Factibles: {n_factibles}/{len(resultados)} ({pct_fact:.1f}%)
╚═══════════════════════════════════════════════════════════╝
    """

    ax_info.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round',
                                                       facecolor='wheat', alpha=0.5))

    fig.suptitle('ALGORITMO GENÉTICO CON PENALIZACIÓN EXTERIOR',
                 fontsize=16, fontweight='bold')

    plt.pause(0.05)


# ALGORITMO GENÉTICO PRINCIPAL
def algoritmo_genetico_penalizacion(tam_poblacion=30, num_generaciones=50,
                                    pc=0.8, pm=0.2, lambda_p=1000, precision=5,
                                    ajuste_adaptativo=True, velocidad_animacion=0.5):
    """Ejecuta el algoritmo genético con penalización exterior para el problema dado.
    Parámetros:
        ajuste_adaptativo: Si True, aumenta λ_P si no hay soluciones factibles
        velocidad_animacion: Pausa entre generaciones en segundos (0.1=rápido, 1.0=lento)
    """
    # Parámetros del problema
    limites = [(0, 1), (0, 2)]  # x ∈ [0,1], y ∈ [0,2]

    print("=" * 70)
    print("ALGORITMO GENÉTICO CON PENALIZACIÓN EXTERIOR")
    print("=" * 70)
    print(f"Problema: min f(x,y) = 4(x-3)² + 3(y-3)²")
    print(f"Sujeto a: g1(x,y) = 2x + y - 2 ≤ 0")
    print(f"          g2(x,y) = 3x + 4y - 6 ≤ 0")
    print(f"          x ∈ [0,1], y ∈ [0,2]")
    print(f"\nParámetros: Np={tam_poblacion}, Gen={num_generaciones}, ")
    print(f"            Pc={pc}, Pm={pm}, λ_P inicial={lambda_p}")
    print(f"            Ajuste adaptativo: {'Activado' if ajuste_adaptativo else 'Desactivado'}")
    print(f"            Velocidad: {velocidad_animacion}s entre generaciones")
    print("=" * 70)
    print("\n🎬 Iniciando visualización en tiempo real...")
    print("   (Cierra la ventana para terminar o espera a que complete)\n")

    # 1. Generar población inicial
    poblacion, longitudes_bits = generar_poblacion_inicial(tam_poblacion, limites, precision)

    # Historial
    historial = {
        'Fp': [],
        'f': [],
        'P': [],
        'pct_factibles': []
    }
    mejor_global = None
    mejor_fitness_global = float('inf')

    # Preparar visualización
    fig, ax_espacio, ax_fitness, ax_valores, ax_factibles, ax_info = preparar_visualizacion()
    axes = (ax_espacio, ax_fitness, ax_valores, ax_factibles, ax_info)

    # Ciclo evolutivo
    for gen in range(num_generaciones):
        # 2. Evaluar población en Fp
        resultados = evaluar_poblacion_penalizada(poblacion, longitudes_bits, limites, lambda_p)
        aptitudes = [r['Fp'] for r in resultados]

        # 3. Selección del mejor
        mejor_actual = encontrar_mejor_individuo(resultados)

        # Actualizar mejor global
        if mejor_actual['Fp'] < mejor_fitness_global:
            mejor_fitness_global = mejor_actual['Fp']
            mejor_global = mejor_actual.copy()

        # Guardar historial
        n_factibles = sum(1 for r in resultados if r['factible'])
        pct_factibles = (n_factibles / tam_poblacion) * 100

        historial['Fp'].append(mejor_actual['Fp'])
        historial['f'].append(mejor_actual['f'])
        historial['P'].append(mejor_actual['P'])
        historial['pct_factibles'].append(pct_factibles)

        # Mostrar progreso en consola cada 5 generaciones
        if gen % 5 == 0:
            print(f"Gen {gen:3d}/{num_generaciones} | Fp={mejor_actual['Fp']:8.4f} | "
                  f"Factibles: {n_factibles:2d}/{tam_poblacion} ({pct_factibles:5.1f}%) | "
                  f"λ_P={lambda_p}")

        # Ajuste adaptativo de λ_P (cada 10 generaciones)
        if ajuste_adaptativo and gen > 0 and gen % 10 == 0:
            if n_factibles == 0:
                lambda_p *= 2  # Duplicar penalización si no hay factibles
                print(f"  ⚠️  [Gen {gen}] No hay factibles. Aumentando λ_P a {lambda_p}")

        # Actualizar visualización
        actualizar_visualizacion(fig, axes, gen, resultados, historial, mejor_global, lambda_p)
        plt.pause(velocidad_animacion)  # Pausa ajustable entre generaciones

        # 4. Selección de padres por torneo
        padres = seleccion_torneo(poblacion, aptitudes, tam_poblacion)
        aptitudes_padres = [evaluar_poblacion_penalizada([p], longitudes_bits, limites, lambda_p)[0]['Fp']
                            for p in padres]

        # 5. Cruzamiento en dos puntos
        hijos = cruzamiento_dos_puntos(padres, pc)

        # 6. Mutación simple
        hijos = mutacion_simple(hijos, pm)

        # 7. Evaluación de descendientes en Fp
        resultados_hijos = evaluar_poblacion_penalizada(hijos, longitudes_bits, limites, lambda_p)
        aptitudes_hijos = [r['Fp'] for r in resultados_hijos]

        # 8. Sustitución por familia
        poblacion, aptitudes = sustitucion_por_familia(padres, aptitudes_padres, hijos, aptitudes_hijos)

    plt.show()

    # Resultados finales
    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print(f"λ_P final utilizado: {lambda_p}")
    print(f"\nMejor solución encontrada:")
    print(f"  x = {mejor_global['x']:.6f}")
    print(f"  y = {mejor_global['y']:.6f}")
    print(f"\nValores:")
    print(f"  f(x,y) = {mejor_global['f']:.6f}")
    print(f"  P(x,y) = {mejor_global['P']:.6f}")
    print(f"  F_P(x,y) = {mejor_global['Fp']:.6f}")
    print(f"\nRestricciones:")
    g1_val = restriccion_g1(mejor_global['x'], mejor_global['y'])
    g2_val = restriccion_g2(mejor_global['x'], mejor_global['y'])
    print(f"  g1(x,y) = {g1_val:.6f} {'✓' if g1_val <= 0 else '✗'}")
    print(f"  g2(x,y) = {g2_val:.6f} {'✓' if g2_val <= 0 else '✗'}")
    print(f"\nEstado: {'✓ SOLUCIÓN FACTIBLE' if mejor_global['factible'] else '✗ SOLUCIÓN NO FACTIBLE'}")

    if not mejor_global['factible']:
        print(f"\n⚠️  ADVERTENCIA: La solución viola restricciones.")
        print(f"   Sugerencia: Aumentar λ_P o más generaciones.")

    print("=" * 70)

    return mejor_global, historial


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # ============================================================================
    # CONFIGURACIÓN DE VELOCIDAD DE ANIMACIÓN
    # ============================================================================
    # Ajusta este valor para controlar la velocidad de la animación:
    #   0.1 = Muy rápido (difícil de seguir)
    #   0.3 = Rápido
    #   0.5 = Normal (RECOMENDADO para ver bien el proceso)
    #   1.0 = Lento (bueno para presentaciones)
    #   2.0 = Muy lento (para análisis detallado)
    # ============================================================================
    VELOCIDAD = 1  # <-- CAMBIA ESTE VALOR para ajustar velocidad
    # ============================================================================

    # Ejecutar AG con Penalización Exterior
    # NOTA: λ_P inicial = 1000 (aumentado desde 100 para garantizar factibilidad)
    # El ajuste adaptativo duplica λ_P si no hay soluciones factibles

    print("=" * 70)
    print("CONFIGURACIÓN DE VELOCIDAD DE ANIMACIÓN")
    print("=" * 70)
    print(f"Velocidad seleccionada: {VELOCIDAD}s entre generaciones")
    print("\nVelocidades disponibles:")
    print("  0.1 = Muy rápido")
    print("  0.3 = Rápido")
    print("  0.5 = Normal (recomendado) ← ACTUAL" if VELOCIDAD == 0.5 else "  0.5 = Normal (recomendado)")
    print("  1.0 = Lento" + (" ← ACTUAL" if VELOCIDAD == 1.0 else ""))
    print("  2.0 = Muy lento" + (" ← ACTUAL" if VELOCIDAD == 2.0 else ""))
    print("=" * 70)

    mejor_solucion, historial = algoritmo_genetico_penalizacion(
        tam_poblacion=30,
        num_generaciones=50,
        pc=0.8,
        pm=0.2,
        lambda_p=1000,  # Aumentado de 100 a 1000
        precision=5,
        ajuste_adaptativo=True,  # Ajusta λ_P automáticamente si es necesario
        velocidad_animacion=VELOCIDAD  # Velocidad de animación ajustable
    )