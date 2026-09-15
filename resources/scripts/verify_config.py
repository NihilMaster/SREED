#!/usr/bin/env python3
"""
Script para verificar la configuración de SREED
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.infrastructure.config import load_settings


def verify_configuration():
    print("=" * 60)
    print("VERIFICACIÓN DE CONFIGURACIÓN DE SREED")
    print("=" * 60)
    
    try:
        settings = load_settings(ROOT)
        
        print("\n✓ Configuración cargada exitosamente\n")
        
        # Verificar rutas
        print("RUTAS DE NEGOCIO:")
        print(f"  Input:     {settings.input_dir} {'✓' if settings.input_dir.exists() else '✗'}")
        print(f"  Processed: {settings.processed_dir} {'✓' if settings.processed_dir.exists() else '✗'}")
        print(f"  Failed:    {settings.failed_dir} {'✓' if settings.failed_dir.exists() else '✗'}")
        
        print("\nRUTAS TÉCNICAS:")
        print(f"  Chroma:    {settings.chroma_dir} {'✓' if settings.chroma_dir.exists() else '✗'}")
        print(f"  Logs:      {settings.log_dir} {'✓' if settings.log_dir.exists() else '✗'}")
        print(f"  Models:    {settings.models_dir} {'✓' if settings.models_dir.exists() else '✗'}")
        
        print("\nCONFIGURACIÓN OLLAMA:")
        print(f"  Host:            {settings.ollama_host}")
        print(f"  Modelo principal: {settings.primary_model}")
        print(f"  Modelo fallback:  {settings.fallback_model}")
        print(f"  Contexto:         {settings.context_window} tokens")
        print(f"  Timeout:          {settings.llm_timeout}s")
        
        print("\nCONFIGURACIÓN OCR:")
        print(f"  Idiomas:  {', '.join(settings.ocr_languages)}")
        print(f"  GPU:      {'Sí' if settings.use_gpu else 'No'}")
        print(f"  PDF DPI:  {settings.pdf_dpi}")
        
        print("\nCONFIGURACIÓN RAG:")
        print(f"  Colección: {settings.chroma_collection}")
        print(f"  Top K:     {settings.rag_top_k}")
        
        print("\nMODO DE OPERACIÓN:")
        print(f"  Modo:      {settings.llm_mode}")
        
        if settings.llm_mode == "qwen_gemini":
            print(f"  Gemini:    {'✓ API Key configurada' if settings.gemini_api_key else '✗ API Key faltante'}")
        else:
            print(f"  Gemini:    No requerido (modo qwen_only)")
        
        print("\n" + "=" * 60)
        print("CONFIGURACIÓN VÁLIDA ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error en configuración: {e}")
        print("\n" + "=" * 60)
        print("CONFIGURACIÓN INVÁLIDA ✗")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = verify_configuration()
    sys.exit(0 if success else 1)