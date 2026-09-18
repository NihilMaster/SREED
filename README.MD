# SREED - Sistema Resiliente de Extracción Estructurada de Documentos

Sistema local de procesamiento automático de facturas escaneadas o digitales, diseñado para entornos con documentos de calidad variable (arrugados, rotos, con sellos superpuestos, en español e inglés). Combina visión por computadora clásica, modelos de lenguaje locales y visión multimodal en la nube para producir extracciones estructuradas con validación determinística.

## Características principales

- Pipeline OCR multi-paso con selección automática de la mejor pasada (imagen original, realzada con CLAHE + upscale, o binarizada con OTSU).
- Decodificación QR multi-estrategia: pyzbar + detector nativo de OpenCV sobre 6 variantes de la imagen (escalas y umbrales).
- Extracción del código fiscal CUFE desde la imagen (vía Gemini Vision) y construcción automática del enlace de validación DIAN.
- Estrategia híbrida de LLMs: Qwen 3.5:9b como modelo principal local, Qwen 2.5:7b como fallback local, y Gemini Flash Vision para corrección cuando la confianza del OCR es baja.
- Reintentos exponenciales automáticos ante saturación de la API de Google (errores 503).
- JSON válido garantizado por gramática (`format="json"`) y modo de pensamiento desactivado en Qwen para evitar respuestas vacías.
- Validación determinística post-extracción: coherencia subtotal + impuestos = total, suma de items, presencia de campos críticos.
- Monitor de carpeta con `watchdog` que procesa archivos automáticamente al llegar.
- Interfaz Streamlit con seguimiento de progreso en tiempo real y JSON copiable.
- Persistencia vectorial en ChromaDB preparada para RAG (pendiente de reintegrar el chat).
- Arquitectura hexagonal (puertos y adaptadores) para permitir sustituir cualquier componente sin tocar el dominio.

## Stack tecnológico

- Python 3.12
- OpenCV + PyMuPDF: preprocesamiento de imágenes y conversión de PDFs.
- EasyOCR: extracción de texto con Deep Learning.
- pyzbar + cv2.QRCodeDetector: decodificación de códigos QR.
- Ollama: ejecución local de modelos Qwen (privacidad + latencia baja).
- Google Gemini API (SDK `google-genai`): visión multimodal para documentos de baja calidad.
- ChromaDB: base de datos vectorial local para embeddings.
- Streamlit: interfaz web local.
- watchdog: monitor de carpeta en segundo plano.
- Tenacity: reintentos exponenciales para servicios externos.

## Estructura del repositorio

``` bash
repo/SREED/
├── invoices/
│   ├── input/          Carpeta monitoreada por watchdog
│   ├── processed/      Documentos exitosos + JSON lateral
│   ├── failed/         Documentos con error
│   └── review/         (pendiente) cola de revisión manual
├── resources/
│   ├── chroma/         Base de datos vectorial
│   ├── logs/           Logs persistentes del monitor
│   ├── models/         Modelfiles personalizados de Ollama
│   └── scripts/        Utilidades
├── src/
│   ├── application/
│   │   ├── ports/      Contratos (DocumentProcessor, LLMProvider, etc.)
│   │   └── use_cases/  ProcessDocument, IndexDocument, AnswerQuestion
│   ├── domain/
│   │   ├── models.py   DocumentResult, DocumentStatus, ChatMessage
│   │   └── validation.py   Validaciones determinísticas
│   ├── infrastructure/adapters/
│   │   ├── chroma/     ChromaVectorStore
│   │   ├── filesystem/ LocalFileSystemAdapter
│   │   ├── llm/        QwenOllamaProvider, GeminiVisionProvider, QwenGeminiStrategy
│   │   ├── ocr/        OpenCV, EasyOCR, pyzbar, PyMuPDF, OCRRealDocumentProcessor
│   │   └── rpa/        WatchdogAdapter
│   └── interface/      monitor_rpa.py, streamlit_app.py
├── orchestrator.py     Arranque unificado de todos los servicios
├── .env                Variables de entorno
└── requirements.txt
```

## Pipeline de procesamiento

Para cada documento que entra a `invoices/input/`:

1. **Watchdog** detecta el archivo y espera 1 segundo de estabilidad (debounce).
2. **OCR multi-paso** ejecuta EasyOCR sobre tres variantes de la imagen y conserva la de mayor confianza:
   - Original
   - Realzada (upscale 3x + CLAHE + escala de grises)
   - Binarizada (OTSU)
3. **Decodificador QR** prueba pyzbar y cv2.QRCodeDetector sobre 6 variantes de la imagen.
4. **Qwen 3.5:9b** extrae campos clave del texto OCR en ~15-80s con JSON garantizado por gramática.
5. Si la confianza del OCR es menor a 0.6, se activa **Gemini Vision**:
   - Reintenta hasta 3 veces con backoff exponencial si Google devuelve 503.
   - Extrae esquema completo (incluyendo items, CUFE y observaciones).
   - El resultado se fusiona con el de Qwen: Gemini gana en campos críticos (total, impuestos, subtotal, NIT, proveedor, items, CUFE).
6. **Validación determinística** verifica coherencia matemática y presencia de campos.
7. El documento se mueve a `processed/` o `failed/` junto con su JSON lateral.
8. **ChromaDB** indexa el documento para futuras consultas RAG.

## Configuración

Variables principales en `.env`:

```env
SREED_PRIMARY_MODEL=qwen3.5:9b
SREED_FALLBACK_MODEL=qwen2.5:7b
SREED_GEMINI_API_KEY=<tu_clave>
SREED_GEMINI_MODEL=gemini-3.6-flash
SREED_LLM_MODE=qwen_gemini
SREED_LLM_TIMEOUT=1800
SREED_OCR_LANGUAGES=es,en
SREED_USE_GPU=true
SREED_CHROMA_COLLECTION=sreed_documents
SREED_RAG_TOP_K=5
```

## Uso

### Opción 1: Orquestador unificado (recomendado)

```bash
python orchestrator.py
```

Arranca Ollama (si no corre), asegura los modelos, inicia el monitor y Streamlit en segundo plano.

### Opción 2: Componentes por separado

```bash
# Terminal 1: monitor en background
python -m src.interface.monitor_rpa

# Terminal 2: interfaz web
streamlit run src/interface/streamlit_app.py
```

### Opción 3: Solo monitor (headless)

```bash
python -m src.interface.monitor_rpa
```

Copia archivos a `invoices/input/` y serán procesados automáticamente.

## Estrategia híbrida de LLMs

| Rol | Modelo | Ejecución | Tarea |
|-----|--------|-----------|-------|
| Principal | Qwen 3.5:9b | Local (Ollama) | Extraer campos clave del texto OCR (tarea corta, ~15-80s) |
| Fallback | Qwen 2.5:7b | Local (Ollama) | Asumir si el principal falla 3 veces seguidas |
| Visión | Gemini 3.6 Flash | Nube | Corregir y completar cuando OCR confidence < 0.6 |

El sistema nunca depende 100% de la nube: si Gemini está saturado (error 503), el resultado de Qwen se conserva y el documento se procesa igualmente.
