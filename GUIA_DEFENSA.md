# Guía de exposición y preguntas cruzadas

**Equipo:** Nicolás Fernández Vera, Bastian Gutierrez y Victor Gutierrez. **Docente:** Jazna Hidalgo. **Sección:** 001D.

## Guion de 10 minutos

Distribución propuesta, ajustable según el trabajo real. Las notas de cada diapositiva contienen recordatorios y fuentes. Todos deben estudiar el notebook completo: la pauta permite preguntas cruzadas.

| Diapositiva | Responsable propuesto | Tiempo | Mensaje central |
|---|---|---:|---|
| 1. Portada | Nicolás | 40 s | Continuidad del caso, objetivo de comparar CNN y MLP |
| 2. Caso | Nicolás | 45 s | Seis letras ASL estáticas, no traducción completa |
| 3. Datos | Nicolás | 50 s | Particiones idénticas, normalización y test separado |
| 4. Arquitectura | Bastian | 65 s | Dos convoluciones, dos pooling y clasificación final |
| 5. Entrenamiento | Bastian | 55 s | Protocolo común y lectura de pérdidas |
| 6. Resultados | Bastian | 65 s | 97,27% vs 87,81%, mejora de 9,46 pp |
| 7. Recursos | Bastian | 45 s | Menos parámetros no garantiza más velocidad |
| 8. Matriz de confusión | Victor | 60 s | 49 errores, concentrados en O y K |
| 9. Ejemplos | Victor | 45 s | Una probabilidad alta puede equivocarse |
| 10. Feature Maps | Victor | 55 s | Profundidad y combinación de patrones |
| 11. Grad-CAM | Victor | 55 s | Mapas por clase, respuestas al fondo y límites |
| 12. Conclusiones | Nicolás | 20 s | Mejora medida y necesidad de validación externa |
| **Total** | | **600 s** | **10 minutos** |

No leer cada número de las tablas. Explicar el significado de una métrica y señalar el resultado relevante. Ensayar con cronómetro y dejar preparado el notebook por si se pide mostrar código.

## Preguntas que cualquiera debe responder

**1. ¿Qué problema resuelven?** Clasificar una imagen de mano 28×28 en una de seis letras estáticas ASL: A, C, F, K, O o Y. Es supervisado, multiclase y de imágenes.

**2. ¿Por qué conservar las seis letras?** Para mantener la variante de E1 y comparar las arquitecturas sobre el mismo problema y las mismas muestras. Cambiar las clases mezclaría el efecto de la arquitectura con el de la dificultad.

**3. ¿Cuáles son las etiquetas originales y nuevas?** A: 0→0, C: 2→1, F: 5→2, K: 10→3, O: 14→4 y Y: 24→5.

**4. ¿Por qué falta J y Z?** El dataset representa letras estáticas. J y Z requieren movimiento. Reconocerlas exigiría información temporal.

**5. ¿Qué cambia entre la entrada MLP y CNN?** El MLP recibe 784 valores. La CNN recibe 28×28×1, manteniendo la organización espacial. El último 1 es el canal de gris. No cambia la cantidad de píxeles.

**6. ¿Aplanar destruye la información?** No elimina valores. El MLP podría aprender relaciones entre posiciones, pero no incorpora de forma explícita localidad y compartición espacial de pesos como la CNN.

**7. ¿Por qué dividir por 255?** Lleva los píxeles a [0,1], manteniendo intensidades relativas. Se hace una vez tanto en train como en validación y test. No se estima ningún parámetro con test.

**8. ¿Cuántos datos usa cada partición?** 5.521 train, 1.381 validación y 1.797 test. La validación es el 20% estratificado de las 6.902 muestras filtradas del train oficial.

**9. ¿Para qué sirve validación?** Monitorear el aprendizaje con datos que no actualizan pesos. Test se reserva para la evaluación final. No usamos test como validación ni para escoger arquitectura.

**10. ¿Qué hace Conv2D?** Desliza filtros aprendidos sobre vecindades. Cada filtro comparte pesos en diferentes posiciones y genera un mapa de activación. Se aprenden por retropropagación, no se diseñan manualmente como detectores fijos de dedos.

**11. ¿Por qué filtros 3×3?** Captan patrones locales con pocos parámetros. Dos bloques permiten combinar patrones sin usar filtros grandes desde el inicio.

**12. ¿Cuántas convoluciones hay?** Dos: 16 filtros y 32 filtros. No hay 48 capas: 16 y 32 son cantidades de filtros dentro de cada capa.

**13. ¿Cuántos pooling hay y qué hacen?** Dos MaxPooling2D 2×2, uno después de cada convolución. Retienen la activación máxima local y reducen dimensiones. No entrenan pesos ni garantizan invariancia total. No crean automáticamente “más características”: reducen mapas y conservan respuestas dominantes.

**14. ¿Por qué 28 pasa a 26?** Con kernel 3, stride 1 y padding valid: salida = floor((28−3)/1)+1 = 26. Pooling 2/stride 2 da 13. Otra convolución da 11, y el pooling final da 5.

**15. ¿Qué hace ReLU?** f(x)=max(0,x). Introduce no linealidad. Su derivada es 1 en positivos y 0 en negativos; en cero se usa una convención. No es el optimizador. Los gradientes se propagan por la red y Adam actualiza pesos.

**16. ¿Qué ocurre si se mantienen valores negativos?** Depende de la activación. Sin activación, apilar operaciones lineales no crea la misma capacidad no lineal. Una activación distinta, como Leaky ReLU, permite valores negativos con otra regla, pero no es la usada aquí.

**17. ¿Qué hace Softmax?** Convierte seis logits en probabilidades no negativas que suman 1. Elegimos la mayor. La cifra no representa automáticamente confianza calibrada y no permite detectar clases desconocidas.

**18. ¿Por qué sparse categorical crossentropy?** Porque tenemos seis categorías exclusivas y etiquetas enteras 0–5. La versión categorical usaría vectores one-hot.

**19. ¿Qué hace Adam?** Usa estimaciones móviles de gradientes y sus cuadrados para adaptar actualizaciones por parámetro. Usamos learning rate inicial 0,001. No asegura siempre ser mejor que cualquier otro optimizador.

**20. ¿Qué son época y batch?** Época: pasada por todos los ejemplos train. Batch: lote utilizado en una actualización. Con 5.521 imágenes y batch 32 se tienen 173 lotes por época, incluyendo el último incompleto.

**21. ¿Cómo calculan parámetros?** Conv: (alto×ancho×canales_entrada + 1 sesgo)×filtros. Dense: (entradas+1)×salidas. CNN total: 160+4.640+51.264+390=56.454. MLP: 100.480+8.256+390=109.126.

**22. ¿Por qué la CNN tarda más teniendo menos parámetros?** Los filtros se aplican muchas veces a diferentes posiciones. El número de parámetros mide pesos almacenados, no todas las operaciones. Aquí CNN tarda 27,00 s y MLP 23,85 s en la misma CPU.

**23. ¿Cuál es el resultado principal?** CNN 97,27% y F1 macro 0,9706, frente a MLP actual 87,81% y 0,8780. Son +9,46 puntos porcentuales. Los errores bajan de 219 a 49, reducción relativa de 77,63%.

**24. ¿Por qué antes decía 87,03%?** Esa es la ejecución histórica de E1. En E2 reentrenamos el mismo MLP en un entorno común con la CNN. Diferentes entornos y mecanismos de barajado pueden cambiar resultados aunque se use semilla 42. No reemplazamos el registro histórico.

**25. ¿Diferencia entre accuracy, precision y recall?** Accuracy mide aciertos globales. Precision de O pregunta cuántas predicciones O realmente son O. Recall de O pregunta cuántas O reales detectamos. La CNN tiene precision O=1 y recall O≈0,8618 porque pierde 34 O, pero no asigna O a otras letras.

**26. ¿Qué es F1 macro?** Se calcula F1 por clase como 2PR/(P+R) y se promedian las seis F1 con igual peso. No equivale a calcular F1 de precision macro y recall macro. Weighted pondera por cantidad de imágenes de cada clase.

**27. ¿Cómo se lee la matriz?** Filas reales y columnas predichas. Diagonal=aciertos. Fuera de diagonal=errores. En CNN: 20 O→K, 14 O→F y 15 K→Y, total 49.

**28. ¿Hay sobreajuste con validación 100%?** Esa cifra no basta para descartarlo. Train y validación provienen del mismo archivo original y pueden ser muy parecidos. La CNN baja a 97,27% en test y MLP a 87,81%. Reportamos brechas y no afirmamos independencia por sujeto.

**29. ¿Qué son Feature Maps?** Respuestas de filtros a una entrada. Conv1 conserva detalles locales y Conv2 combina respuestas de regiones más grandes. No son mapas específicos de letra y no tienen un significado humano fijo.

**30. ¿Qué aporta Grad-CAM?** Aproxima qué regiones contribuyen positivamente al logit de una clase usando gradientes respecto a mapas convolucionales. Calculamos uno por letra. Es distinto de mostrar cualquier filtro y no demuestra causalidad.

**31. ¿Los mapas muestran solo dedos?** No. Algunas respuestas, especialmente en C/F, también abarcan el fondo. Es una limitación a investigar, no una evidencia de comprensión perfecta de la mano.

**32. ¿Por qué no usaron Dropout?** La rúbrica penaliza capas de regularización extra con 20%. La entrega utiliza únicamente los componentes solicitados. Grad-CAM no cambia la arquitectura entrenada.

**33. ¿Qué falta para usarlo con cámara?** Detectar y recortar la mano, controlar preprocesamiento, evaluar nuevas personas, fondos e iluminación, rechazar entradas desconocidas y medir latencia del flujo completo. El <10 ms medido solo cubre la llamada al modelo.

**34. ¿Qué asegura reproducibilidad?** Código, CSV, huellas, índices de partición, semillas, versiones, modelos e historiales guardados. No garantizamos igualdad bit a bit entre distintos dispositivos.

**35. ¿Cumplen todos los objetivos de negocio?** Cumplen los umbrales de acierto, F1, error y latencia local en el dataset. No se ha validado accesibilidad con usuarios, traducción real ni uso productivo. No se debe equiparar una métrica de laboratorio con impacto social demostrado.

## Antes de entregar

- Confirmar responsabilidades reales y que los nombres estén correctos.
- Abrir la presentación en el equipo de exposición y ensayar cambios de diapositiva.
- Saber ubicar arquitectura, matriz de confusión y Grad-CAM en el notebook.
- Publicar el repositorio y comprobar acceso para la docente.
- Entregar dentro del plazo anunciado. La pauta penaliza atraso (10%), capas extra de regularización (20%) y ausencia de matriz de responsabilidades (10%).
