# Sistema Resiliente de Extracción Estructurada de Documentos (SREED)

## 1. Resumen Ejecutivo

El SREED es un Producto Mínimo Viable (MVP) diseñado para automatizar la recepción, lectura y estructuración de datos de documentos no estructurados (facturas físicas escaneadas o digitales). El sistema aborda escenarios de alta complejidad, como facturas arrugadas, rotas, con sellos superpuestos y en múltiples idiomas.

Su núcleo combina visión por computadora clásica con modelos de Inteligencia Artificial (LLMs) para ir más allá de un simple OCR, logrando una **extracción estructurada y semántica** de la información. Además, implementa una arquitectura Híbrida/RAG que permite consultar el histórico de documentos procesados mediante lenguaje natural, sentando las bases para una automatización a gran escala.

## 2. Arquitectura y Decisiones Técnicas

El diseño se basa en una arquitectura modular (Separation of Concerns) que permite escalar o reemplazar componentes sin afectar el flujo principal.

### A. Preprocesamiento y Extracción Óptica (OCR)

* **Decisión:** Uso de **OpenCV** para el preprocesamiento, **EasyOCR** para el texto y **pyzbar** para códigos de barras/QR.
* **Justificación:** Se descartó Tesseract OCR porque es altamente sensible al ruido. EasyOCR, al estar basado en Deep Learning, es excepcionalmente robusto frente a documentos rotos, textos inclinados, fuentes no estándar y sellos. OpenCV actúa como un filtro previo para binarizar y normalizar la imagen. `pyzbar` extrae datos implícitos o fiscales ocultos en los QR, aportando una capa de validación extra.

### B. Motor de Extracción Estructurada (IA Híbrida)

* **Decisión:** Orquestación de dos modelos distintos según la tarea.
* *Tarea de Estructuración (JSON):* **API de Google Gemini (1.5 Flash)**.
* *Tarea de RAG/Chat Analítico:* **Qwen2.5-7B-Instruct (Despliegue Local)**.


* **Justificación:** La ingeniera solicitó explícitamente familiaridad con modelos LLM (como "Luna" o GPT) para interpretar los datos. Utilizar Gemini garantiza una estructuración rápida y precisa del texto desordenado en un formato JSON limpio (Proveedor, NIT, Total, etc.) a costo cero. Por otro lado, usar Qwen en local para responder preguntas sobre el histórico de facturas demuestra conocimientos de ciberseguridad y privacidad de datos, asegurando que la analítica de la empresa no dependa 100% de la nube.


* **Resiliencia:** El sistema incluye un *Switch Offline*, permitiendo que el modelo local asuma la tarea de estructuración si la API externa falla.

### C. Automatización de Flujos (RPA Ligero)

* **Decisión:** Implementación de un demonio en segundo plano usando la librería **`watchdog`** en Python.


* **Justificación:** Para cubrir la expectativa de automatización, el sistema no depende exclusivamente de que el usuario cargue el archivo manualmente. El script monitorea activamente una carpeta del sistema operativo. Cuando un sistema de terceros (ej. un cliente de correo) descarga una factura allí, el sistema la procesa automáticamente.

### D. Persistencia y Vectorización

* **Decisión:** Uso de **ChromaDB** como base de datos vectorial local.
* **Justificación:** Para lograr un MVP funcional en pocos días, ChromaDB permite incrustar (embed) los JSON resultantes sin necesidad de levantar servidores ni configurar redes. Demuestra el conocimiento de bases de datos vectoriales necesarias para flujos evolutivos y agentes IA. *Nota de escalabilidad: El código está diseñado con puertos y adaptadores; para producción, ChromaDB se puede sustituir por PostgreSQL con la extensión `pgvector*`.



## 3. Interfaces de Usuario

El usuario final interactuará con el sistema de dos formas complementarias:

### Interfaz 1: Carpeta Activa de Automatización (Background)

* **Descripción:** Una simple carpeta en el sistema operativo (ej. `C:\SREED_Input\`).
* **Uso:** El usuario (o un sistema legacy) simplemente arrastra, copia o guarda los archivos PDF/PNG/JPG en esta carpeta. No hay botones ni clics; el sistema `watchdog` los detecta, los procesa de forma invisible y mueve el archivo a una subcarpeta de `Procesados/`, dejando el JSON resultante listo en la base de datos.

### Interfaz 2: Dashboard Interactivo (Streamlit)

* **Descripción:** Una aplicación web local accesible desde el navegador, con un diseño limpio y corporativo. Se eligió Streamlit sobre Gradio por su capacidad para manejar múltiples vistas y mantener el estado de la sesión.
* **Uso y Vistas:**
* **Pestaña de Procesamiento Manual:** Una zona de *Drag & Drop* para subir una factura específica. Al procesarla, muestra la pantalla dividida: a la izquierda la imagen de la factura con recuadros sobre el texto detectado, y a la derecha el JSON estructurado con los datos extraídos (y un botón para corregirlos manualmente si es necesario).
* **Pestaña de Asistente Analítico (RAG):** Una interfaz de chat. El usuario puede escribir en lenguaje natural: *"Genera un resumen de los impuestos pagados a proveedores de tecnología esta semana"*. El sistema busca en ChromaDB y el modelo Qwen genera la respuesta basándose en los datos extraídos previamente.
* **Configuración:** Un panel lateral para hacer el *switch* entre API Nube (Gemini) y Modelo Local (Qwen).



## 4. Lista de Entregables (Estructura del Repositorio)

El proyecto se entregará en un repositorio Git estructurado de la siguiente manera, listo para su ejecución local:

* 📁 `proyecto_ocr_rag/`
* 📁 `input_facturas/` *(Carpeta monitoreada por el sistema)*.
* 📁 `facturas_procesadas/` *(Carpeta destino para documentos ya leídos)*.
* 📁 `db_vectorial/` *(Directorio autogenerado por ChromaDB para la persistencia)*.
* 📄 `app.py` *(Punto de entrada del Front-end en Streamlit).*
* 📄 `monitor_rpa.py` *(Script demonio para la automatización con Watchdog).*
* 📄 `motor_ocr.py` *(Módulo lógico: OpenCV, EasyOCR, pyzbar).*
* 📄 `extractor_llm.py` *(Módulo de orquestación de LLMs: Prompts para Gemini y Qwen).*
* 📄 `rag_chat.py` *(Módulo de consulta: Embeddings y recuperación con ChromaDB).*
* 📄 `requirements.txt` *(Dependencias estrictas para replicar el entorno: opencv-python, easyocr, streamlit, chromadb, google-generativeai, etc.).*
* 📄 `README.md` *(Documentación técnica, justificación de arquitectura y guía de ejecución).*