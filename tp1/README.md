# Procesamiento de Lenguaje Natural II - CEIA - FIUBA
# TP1 — TinyGPT (Simple + MOE)
# Alumno: Florentino Arias

**Tarea I**: Implementar modificaciones en la función 'generate' para decodificación greedy, muestreo por temperatura y muestreo top-k/top-p.

Configurar el entorno del notebook. 

Implementar la función 'generateV2' con decodificación greedy, temperatura y top-k/top-p. 

**Tarea II**:

Implementar Mixture of Experts (MoE). Completar la clase 'Expert'.

Implementar la clase 'MoELayer' que es el núcleo del Mixture of Experts.

Actualizar configuración para incluir el atributo 'moe' en lugar de 'moe_args'.

Corregir la configuración para usar 'num_experts_per_token=2' en lugar de 1 para tener al menos 2 expertos activos.

Corregir la clase 'MoELayer' para usar 'self.args' correctamente.

Corregir la clase 'MoELayer' para usar el atributo correcto 'moe_args'.

Podemos ver las diferencias entre las estrategias de decodificación:
1. **Greedy (temp=0.0)**: Produce una salida muy repetitiva (muchas 'x')
2. **Temperatura baja (0.5)**: Más conservadora que la original
3. **Temperatura alta (1.5)**: Más aleatoria y variada
4. **Top-k**: Limita las opciones a los 5 tokens más probables
5. **Top-p**: Usa nucleus sampling con 80% de probabilidad acumulada
6. **Combinación**: Combina top-k y top-p para un balance

**Tarea II**: Convertir TinyGPT en un Mixture of Experts (MoE) con al menos 2 expertos.

Configurar el modelo MoE.

El modelo MoE funciona correctamente, podemos ver que tiene significativamente más parámetros (309,000 vs 109,952) debido a los múltiples expertos.

Probamos la generación de texto con el modelo MoE (creamos una función 'generateV2_MoE' que usa el modelo MoE). 

Comparar los resultados entre el modelo original, el modelo con nuestras mejoras de decodificación, y el modelo MoE: 
### **Tarea I - Estrategias de Decodificación** ✅
- **Decodificación Greedy**: Implementada con `temperature=0.0` para obtener salidas deterministas
- **Muestreo por Temperatura**: Control de aleatoriedad con diferentes valores de temperatura
- **Top-k Sampling**: Limita las opciones a los k tokens más probables
- **Top-p (Nucleus) Sampling**: Usa probabilidad acumulada para balancear diversidad y calidad
- **Combinaciones**: Permite usar múltiples técnicas simultáneamente

### **Tarea II - Mixture of Experts (MoE)** ✅
- **Expert**: Redes especializadas (MLPs) que procesan diferentes aspectos de la entrada
- **Gate**: Mecanismo de routing que decide qué expertos usar para cada token
- **MoELayer**: Combina las salidas de múltiples expertos según los pesos del gate
- **Integración**: El modelo MoE se integra perfectamente con la arquitectura TinyGPT
- **Escalabilidad**: 4 expertos con 2 activos por token, aumentando la capacidad del modelo

### **Resultados Destacables**:
- El modelo MoE tiene **2.8x más parámetros** que el original (309K vs 110K)
- Las diferentes estrategias de decodificación ofrecen un **control granular** sobre la generación
- La implementación es **modular y extensible**, permitiendo futuras mejoras
- Los experimentos muestran claras **diferencias en el comportamiento** de cada estrategia
