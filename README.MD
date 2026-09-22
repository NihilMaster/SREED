# SREED — Sistema Resiliente de Extracción Estructurada de Documentos

SREED es un pipeline automatizado y resiliente para la ingesta, procesamiento, estructuración y consulta analítica de facturas electrónicas y tickets de venta. Combina OCR multivariado, modelos de lenguaje locales (Ollama) y en la nube (Google Gemini), validaciones determinísticas y verificación fiscal (DIAN) para garantizar precisión y trazabilidad en documentos de calidad variable.

---

## Características Principales

- **OCR Resiliente Multi-paso**: EasyOCR combinado con preprocesamiento OpenCV (escalado, escala de grises, CLAHE, binarización) para maximizar la extracción en imágenes de baja calidad.
- **Estrategia Híbrida de LLM**: Qwen 3.5:9b local para extracción rápida de campos clave, con Gemini 3.6 Flash Vision como corrector automático cuando la confianza del OCR es baja (< 0.6).
- **Cross-Check Fiscal DIAN**: Parser determinístico que extrae datos del código QR fiscal (NumFac, FecFac, NitFac, ValFac, ValIva, CUFE) con prioridad absoluta sobre el LLM.
- **Validación Matemática Automática**: Detección de inconsistencias (subtotal + impuestos ≠ total) con distinción entre regímenes de IVA incluido (retail) y aditivo (B2B).
- **RAG con FAISS**: Motor de consulta analítica sobre documentos procesados, utilizando FAISS para búsqueda vectorial robusta y `all-MiniLM-L6-v2` para embeddings locales.
- **Arquitectura a Prueba de Fallos**: Reintentos automáticos ([backoff exponencial](https://learn.microsoft.com/es-es/dotnet/architecture/microservices/implement-resilient-applications/implement-retries-exponential-backoff)), fallbacks de modelos y almacenamiento en archivos planos para evitar corrupción de datos.

---

## Arquitectura del Pipeline

El flujo de procesamiento sigue estos pasos para cada documento que ingresa a `invoices/input/`:

1. **Detección**: Watchdog monitorea la carpeta de entrada con [debounce](https://code-garage.com/en/blog/the-concept-of-debounce-what-is-it) y espera de estabilidad del archivo.
2. **Validación**: Se ignoran archivos temporales, ocultos o con extensiones no soportadas. Archivos soportados: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.
3. **Conversión PDF**: Si es PDF, PyMuPDF convierte cada página a imagen PNG con DPI configurable.
4. **OCR Multivariable**: EasyOCR ejecuta múltiples pasadas (original, mejorada, binarizada) y selecciona la de mayor confianza.
5. **Extracción de QR**: Se intenta decodificar códigos QR con `pyzbar` y `cv2.QRCodeDetector` sobre versiones originales, grises, OTSU y escaladas.
6. **Estructuración LLM**: Qwen 3.5:9b extrae campos clave en formato JSON. Si falla consecutivamente, se activa Qwen 2.5:7b como fallback.
7. **Complemento Gemini**: Si `ocr_confidence < 0.6`, Gemini Vision corrige campos críticos directamente sobre la imagen.
8. **Cross-Check DIAN**: Si existe QR fiscal, se parsea y sobrescribe los valores del LLM con datos fiscales firmados.
9. **Validación**: Se normalizan números, se valida coherencia matemática y se registran observaciones.
10. **Persistencia**: El documento se mueve a `processed/` o `failed/` y se genera un JSON lateral con el resultado completo.
11. **Indexación**: El texto estructurado se vectoriza con FAISS y se guarda con metadatos para consultas RAG.

---

## Stack Tecnológico

**Lenguaje y Frameworks**: Python 3.12, python-dotenv, Pydantic, Tenacity, watchdog, Streamlit

**Visión y OCR**: Pillow, OpenCV, NumPy, EasyOCR, pyzbar, PyMuPDF

**Vectorización y RAG**: FAISS CPU, sentence-transformers (all-MiniLM-L6-v2)

**Modelos de Lenguaje**: Ollama (qwen3.5:9b, qwen2.5:7b), google-genai (Gemini 3.6 Flash)

**Utilidades**: Shell / PowerShell para scripts operativos

---

## Estructura del Repositorio

``` bash
SREED/
├── invoices/                          # Carpetas de flujo de trabajo
│   ├── input/                         # Archivos pendientes (monitoreados por watchdog)
│   ├── processed/                     # Documentos procesados exitosamente + JSONs laterales
│   └── failed/                        # Documentos con errores o no soportados
│
├── resources/                         # Recursos persistentes y utilidades
│   ├── faiss/                         # Índice vectorial FAISS y metadatos (documents.json)
│   ├── logs/                          # Logs de ejecución del monitor, Streamlit y LLMs
│   ├── models/                        # Modelfiles personalizados de Ollama
│   └── scripts/                       # Scripts de utilidad
│       ├── diagnose_rag.py            # Diagnóstico de documentos indexados y recuperación
│       ├── reindex_from_sidecars.py   # Reindexación desde JSONs procesados
│       ├── tree.ps1                   # Comando PowerShell para exploración de carpetas
│       └── verify_config.py           # Verificación de configuración y dependencias
│
├── src/                               # Código fuente (Arquitectura Hexagonal)
│   ├── application/                   # Casos de uso y puertos (lógica de negocio)
│   │   ├── ports/                     # Interfaces abstractas
│   │   │   ├── document_processor.py  # Puerto para procesamiento de documentos
│   │   │   ├── file_system.py         # Puerto para operaciones de archivo
│   │   │   ├── image_preprocessor.py  # Puerto para preprocesamiento de imágenes
│   │   │   ├── llm_provider.py        # Puerto para proveedores de LLM
│   │   │   ├── ocr_extractor.py       # Puerto para extractores OCR
│   │   │   ├── qr_extractor.py        # Puerto para extractores de códigos QR
│   │   │   └── vector_store.py        # Puerto para almacenamiento vectorial
│   │   └── use_cases/                 # Casos de uso concretos
│   │       ├── answer_question.py     # Caso de uso para consultas RAG
│   │       ├── index_document.py      # Caso de uso para indexación de documentos
│   │       ├── process_and_index_document.py  # Orquestación de procesamiento + indexación
│   │       └── process_document.py    # Caso de uso para procesamiento de documentos
│   │
│   ├── domain/                        # Modelos de dominio y reglas de negocio
│   │   ├── dian_qr.py                 # Parser de QR fiscal DIAN y cross-check
│   │   ├── models.py                  # Entidades: DocumentResult, ChatMessage, RetrievedDocument
│   │   └── validation.py              # Reglas de validación matemática y de formato
│   │
│   ├── infrastructure/                # Adaptadores e implementaciones concretas
│   │   ├── adapters/                  # Implementaciones de puertos
│   │   │   ├── faiss/                 # Adaptador FAISS para búsqueda vectorial
│   │   │   ├── filesystem/            # Adaptador de sistema de archivos local
│   │   │   ├── llm/                   # Adaptadores de LLMs (Qwen, Gemini, estrategia híbrida)
│   │   │   ├── ocr/                   # Adaptadores OCR (EasyOCR, OpenCV, PyMuPDF, pyzbar)
│   │   │   └── rpa/                   # Adaptador watchdog para monitoreo de carpetas
│   │   └── config.py                  # Carga y validación de configuración desde .env
│   │
│   └── interface/                     # Interfaces de usuario y servicios
│       ├── monitor_rpa.py             # Servicio de monitoreo de carpetas (watchdog)
│       └── streamlit_app.py           # Aplicación web Streamlit (dashboard y chat RAG)
│
├── orchestrator.py                    # Script de orquestación para iniciar todos los servicios
├── .env.example                       # Plantilla de variables de entorno
├── requirements.txt                   # Dependencias Python del proyecto
├── README.MD                          # Este archivo
└── .gitignore                         # Archivos y carpetas ignorados por Git
```

---

## Instalación y Configuración

### Requisitos Previos

- Python 3.12+
- Ollama instalado y corriendo en `http://localhost:11434`
- Modelos Ollama: `qwen3.5:9b` y `qwen2.5:7b`
- API Key de Google Gemini con permisos para `generativelanguage.googleapis.com`

### Pasos de Instalación

1. **Clonar el repositorio y crear entorno virtual**:

   ``` bash
   python -m venv .venv
   ```

   En Windows PowerShell:

   ``` powershell
   .\.venv\Scripts\Activate.ps1
   ```

   En Linux/macOS:

   ``` bash
   source .venv/bin/activate
   ```

2. **Instalar dependencias**:

   ``` bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno**:

   Copia `.env.example` a `.env` y completa los valores:

   ```env
   SREED_PRIMARY_MODEL=qwen3.5:9b
   SREED_FALLBACK_MODEL=qwen2.5:7b
   SREED_GEMINI_API_KEY=tu_api_key_aqui
   SREED_GEMINI_MODEL=gemini-3.6-flash
   SREED_LLM_MODE=qwen_gemini
   SREED_LLM_TIMEOUT=900
   ```

4. **Verificar configuración**:

   ```bash
   python resources/scripts/verify_config.py
   ```

---

## Uso

### Opción Recomendada: Orquestador

El orquestador verifica dependencias, inicia Ollama si es necesario, y lanza todos los servicios:

```bash
python orchestrator.py
```

Accede a la interfaz web en `http://localhost:8501`.

### Ejecución Manual de Servicios

**Monitor de procesamiento** (escucha `invoices/input/`):

```bash
python -m src.interface.monitor_rpa
```

**Interfaz web Streamlit** (dashboard y chat RAG):

```bash
streamlit run src/interface/streamlit_app.py
```

Con el monitor activo, copia archivos a `invoices/input/` para iniciar el flujo automático.

---

## Utilidades y Mantenimiento

### Diagnóstico de RAG

Verifica documentos indexados y recuperación de consultas:

```bash
python resources/scripts/diagnose_rag.py
```

### Reindexación desde Sidecars

Reconstruye el índice FAISS desde los JSONs procesados:

```bash
python resources/scripts/reindex_from_sidecars.py
```

Para reindexación completa (borra índice existente):

```bash
python resources/scripts/reindex_from_sidecars.py --reset
```

---

## Estrategia de LLMs

| Flujo | Modelo / Proveedor | Ejecución | Tarea |
| ------- | ------------------- | ----------- | ------- |
| Extracción principal | Qwen 3.5:9b | Local (Ollama) | Extraer campos estructurados desde OCR |
| Fallback local | Qwen 2.5:7b | Local (Ollama) | Mantener el flujo si Qwen principal falla |
| Corrección visual | Gemini Vision | Nube (Google GenAI) | Complementar OCR de baja confianza, completar CUFE e items |
| RAG / respuesta de consulta | Qwen 2.5:7b | Local (Ollama) | Responder con contexto recuperado en FAISS |

**Notas importantes**:

- La base del sistema es local. Gemini es opcional y se usa solo como complemento para OCR difícil.
- La extracción estructurada exige JSON válido con temperatura baja.
- El RAG responde con texto libre usando solo el contexto recuperado, sin alucinaciones.

---

## Consideraciones de Diseño

SREED está diseñado para entornos de captura heterogénea, donde la calidad de las facturas varía significativamente. El valor del sistema no reside únicamente en el uso de modelos de IA avanzados, sino en la construcción de un pipeline resiliente que:

- **Maneja documentos reales**: OCR con múltiples pasadas, preprocesamiento adaptativo y decodificación QR robusta.
- **Garantiza precisión fiscal**: Cross-check con QR DIAN como fuente autoritativa firmada.
- **Detecta errores automáticamente**: Validación matemática que identifica inconsistencias antes de que lleguen al usuario final.
- **Opera sin dependencia de la nube**: Fallbacks locales y persistencia en archivos planos aseguran continuidad operativa.
- **Proporciona trazabilidad completa**: Cada decisión del sistema está documentada en JSONs laterales con métricas de confianza y observaciones.

Este enfoque prioriza la robustez y la confiabilidad sobre la complejidad innecesaria, haciendo de SREED un MVP de nivel producción listo para entornos empresariales reales.

---
