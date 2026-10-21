# 🎵 MusicGen – Generador de Vallenato con IA

Proyecto que usa el modelo **MusicGen** de Meta (a través de Hugging Face Transformers) para generar música a partir de una descripción en texto. En este caso, el prompt describe un vallenato tradicional colombiano con acordeón, guacharaca y caja vallenata.

El script aprovecha la **GPU NVIDIA (RTX)** si está disponible, y si no, se ejecuta en CPU.

## 📁 Estructura del proyecto

```
musicgen
├─ .python-version        # Versión de Python usada por uv
├─ musica.wav             # Audio generado por el modelo
├─ pyproject.toml         # Dependencias y configuración del proyecto
├─ README.md
├─ src
│  └─ musicgen
│     └─ __init__.py      # Script principal de generación
└─ uv.lock                # Versiones exactas de las dependencias
```

## ⚙️ Requisitos

- Python (versión indicada en `.python-version`)
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes
- (Opcional) GPU NVIDIA con drivers actualizados para usar CUDA

Dependencias principales:

- `torch`
- `transformers`
- `scipy`

## 🚀 Instalación y ejecución

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone <url-del-repo>
   cd musicgen
   ```

2. Instalar las dependencias:

   ```bash
   uv sync
   ```

3. Ejecutar el script:

   ```bash
   uv run python src/musicgen/__init__.py
   ```

Al terminar se genera el archivo `musica.wav` en la raíz del proyecto.

## 🖥️ Uso de la GPU

Para verificar si PyTorch reconoce la GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Si muestra `False`, significa que se instaló PyTorch sin soporte CUDA y el modelo correrá en CPU (funciona, pero bastante más lento). Para que uv descargue la versión con GPU, se agrega esto al `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

Y luego se vuelve a correr `uv sync`.

## 🧠 ¿Cómo funciona?

1. **Carga del modelo:** se descarga `facebook/musicgen-small` desde Hugging Face junto con su procesador.
2. **Selección de dispositivo:** si hay GPU disponible se usa `cuda` con precisión `float16` para que sea más rápido y gaste menos memoria; si no, se usa la CPU.
3. **Procesamiento del prompt:** el texto se convierte en tokens que el modelo entiende.
4. **Generación:** el modelo produce tokens de audio. Se usa `do_sample=True` para que cada resultado sea diferente y `guidance_scale=3.0` para que el audio se ajuste mejor al texto.
5. **Exportación:** los tokens se convierten en una onda de audio y se guardan como `.wav` con `scipy`.

## 🎛️ Parámetros que se pueden cambiar

| Parámetro | Descripción |
|---|---|
| `PROMPT` | Descripción de la música que se quiere generar |
| `MODEL_ID` | Modelo a usar (`musicgen-small`, `musicgen-medium`, `musicgen-large`) |
| `max_new_tokens` | Duración del audio (50 tokens ≈ 1 segundo, máximo recomendado ~1500 ≈ 30 s) |
| `guidance_scale` | Qué tanto se apega el resultado al prompt |

## ⚠️ Limitaciones

- MusicGen está entrenado para clips de máximo unos 30 segundos; si se pide más, el audio pierde calidad o se vuelve repetitivo.
- El modelo no conoce muy bien géneros regionales como el vallenato, así que el resultado es una aproximación y no siempre suena auténtico.
- La primera ejecución tarda más porque se descarga el modelo.

## 👤 Autor

**Darío Sebastián Rivera Sáenz**