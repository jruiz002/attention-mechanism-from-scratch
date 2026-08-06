# Mecanismo de Atencion desde cero — Laboratorio Semana 5

Implementacion **desde cero** (solo tensores de PyTorch, sin `autograd` ni `nn.Module`) de un
Seq2Seq LSTM EN→ES con **atencion de producto punto escalado**: forward, backward manual y
descenso de gradiente, mas la visualizacion de los mapas de atencion.

Curso: Deep Learning — UVG.

---

## Que hay aqui

| Archivo | Contenido |
|---|---|
| [S5 - Lab_Semana5_Atencion.ipynb](S5%20-%20Lab_Semana5_Atencion.ipynb) | Notebook completo y ejecutado (Bloques 0–12 + preguntas de analisis) |
| [outputs/](outputs/) | Figuras generadas por el notebook |
| `.gitignore` | Ignora artefactos de Python/Jupyter; las figuras de `outputs/` si se versionan |

## Como ejecutar

```bash
pip install torch numpy matplotlib jupyter
jupyter notebook "S5 - Lab_Semana5_Atencion.ipynb"
```

Ejecutar todas las celdas en orden (`Kernel → Restart & Run All`). Tiempo total: ~1 minuto en CPU
(incluye el experimento opcional de 100 iteraciones). El notebook escribe sus figuras en `outputs/`,
que se crea automaticamente.

Sin dependencias de GPU: todo corre en CPU con tensores pequenos (`d_emb=16`, `d_hid=32`, `d_k=d_v=16`).

## Arquitectura implementada

```
x_1..x_T ──► Encoder LSTM ──► H_enc (T, d_hid) ──┬──► K = H_enc W_K^T
                                  │              └──► V = H_enc W_V^T
                                  └── h_T ──► estado inicial del decoder
                                                 │
paso s del decoder:  h_s^dec ──► q_s = W_Q h_s^dec
                     e_s = K q_s / sqrt(d_k)  ──►  alpha_s = softmax(e_s)
                     c~_s = V^T alpha_s
                     z_s = W_out [h_s^dec ; c~_s] + b_out
```

El vector de contexto **fijo** de la Semana 4 ($c = h_T^{enc}$) se reemplaza por uno **dinamico**
$\tilde{c}_s$ recalculado en cada paso del decoder.

## Bloques del laboratorio

| Bloque | Contenido | Responsable |
|---|---|---|
| 0 | Corpus, vocabularios, pesos, celda LSTM (dado) | — |
| 1–4 | Proyecciones Q/K/V, scores, softmax, contexto dinamico | Integrante 1 (+ Pregunta 1) |
| 5–6 | Forward completo del decoder con atencion; backward de la atencion | Gerardo (+ Pregunta 2) |
| 7–10 | Backward del encoder, actualizacion de parametros, loop de entrenamiento, visualizaciones | Jose Auyon (+ Pregunta 3) |
| 11 | Preguntas de analisis | Los tres |
| 12 | Nota automatica de la seccion de codigo | — |

Agregados sobre el enunciado base:
- **Baseline sin atencion** (seq2seq de la Semana 4) entrenado en paralelo para la comparacion de convergencia (`losses_no_attn`).
- **Verificacion con `torch.autograd`** de todos los gradientes manuales (Bloques 6 y 7).
- **Mapa de atencion con teacher forcing**, porque tras 5 iteraciones el decode greedy todavia colapsa en `<EOS>`.
- **Experimento de 100 iteraciones** midiendo la entropia de $\alpha$ iteracion a iteracion (evidencia para la Pregunta 2b).

## Resultados

**Convergencia (5 iteraciones sobre 58 pares de entrenamiento, SGD por par, lr = 0.01):**

| Iteracion | Con atencion | Sin atencion |
|---|---|---|
| 1 | 5.0725 | 5.0730 |
| 2 | 5.0282 | 5.0301 |
| 3 | 4.9842 | 4.9875 |
| 4 | 4.9405 | 4.9452 |
| 5 | **4.8970** | 4.9032 |

Reduccion: **3.46 %** con atencion vs 3.35 % sin atencion. La ventaja es real pero pequena a 5
iteraciones: las oraciones del corpus tienen 2–5 tokens, asi que el cuello de botella del contexto
fijo apenas penaliza al baseline.

**Hallazgo principal — la atencion no se concentra en este regimen.** Tras 100 iteraciones la loss
baja 26 % (5.0725 → 3.7497) pero la entropia de $\alpha$ se queda en $\log 3 = 1.0986$ nats, o sea
pesos uniformes. La razon esta en las escalas: con pesos inicializados a $0.1\cdot\mathcal{N}(0,1)$
los hidden states valen $\sim 8.5\times10^{-3}$, los scores $\sim 10^{-5}$, y el softmax de valores
casi identicos es exactamente uniforme. Con $\alpha$ uniforme el jacobiano del softmax anula la ruta
de los scores: $\|\partial L/\partial W_V\|_\infty \approx 2\times10^{-3}$ contra
$\|\partial L/\partial W_Q\|_\infty \approx 3.5\times10^{-8}$. La **Ruta 1 (values) aprende y la Ruta 2
(scores) no**: el modelo usa la atencion como un promedio fijo de los values, no como un selector.
Para romperlo haria falta una inicializacion mayor ($\mathcal{N}(0,1/\sqrt{d})$), un learning rate
propio para $W_Q$/$W_K$, o normalizar los hidden states antes de proyectarlos.

**Figuras** (`outputs/`):

| Archivo | Que muestra |
|---|---|
| `convergencia_atencion.png` | Loss con atencion vs sin atencion (5 iteraciones) |
| `mapa_atencion.png` | Mapa de atencion con decode greedy |
| `mapa_atencion_teacher_forcing.png` | Mapa de `she sings well → canta bien` con teacher forcing |
| `convergencia_100_iteraciones.png` | Loss y entropia de $\alpha$ a lo largo de 100 iteraciones |
| `mapa_atencion_100_iteraciones.png` | Mapa de atencion tras 100 iteraciones |

## Verificaciones

El autograder del notebook da **55/60** en la seccion de codigo. Los Bloques 1–6 y 9 pasan.

El Bloque 8 (`W_out_new`) reporta hash incorrecto pese a aplicar la regla estandar
$\theta \leftarrow \theta - \alpha\,\partial L/\partial\theta$. Se dejo la version matematicamente
correcta y se anadio una celda que compara **todos** los gradientes manuales contra `torch.autograd`
sobre el mismo forward:

```
dW_Q     max|autograd - manual| = 3.553e-15
dW_K     max|autograd - manual| = 3.553e-15
dW_V     max|autograd - manual| = 2.328e-10
dW_out   max|autograd - manual| = 9.313e-10
db_out   max|autograd - manual| = 2.980e-08
dH_enc   max|autograd - manual| = 3.725e-09
```

Las diferencias son ruido de punto flotante en `float32`, asi que `dW_out` es correcto y por lo tanto
tambien `W_out_new`. Ademas, ningun multiplo escalar de `dW_out` (barrido de $k \in [-5, 5]$) reproduce
el hash esperado, lo que descarta que la diferencia sea un factor de escala (promediar o no sobre $S$,
otro learning rate). Nota aparte: los hashes de `W_Q_new` y `W_K_new` pasan trivialmente, porque
$\partial L/\partial W_Q \sim 10^{-8}$ se redondea a cero con los 5 decimales que usa `_hash_tensor`.
