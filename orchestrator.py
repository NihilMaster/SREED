import subprocess
import sys
import signal
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SREEDOrchestrator:
    """Orquestador para arrancar todos los servicios de SREED"""

    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Registrar manejador de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manejador de señales de parada"""
        logger.info("\nDeteniendo SREED...")
        self.stop_all()
        sys.exit(0)
    
    def check_ollama(self) -> bool:
        """Verifica si Ollama está corriendo"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def start_ollama(self):
        """Inicia Ollama en segundo plano"""
        if self.check_ollama():
            logger.info("Ollama ya está corriendo")
            return
        
        logger.info("Iniciando Ollama...")
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.processes.append(("ollama", process))
        
        # Esperar a que Ollama esté listo
        for i in range(10):
            if self.check_ollama():
                logger.info("Ollama iniciado correctamente")
                return
            time.sleep(1)
        
        logger.warning("Ollama no respondió, continuando de todas formas")
    
    def ensure_models(self):
        """Asegura que los modelos necesarios estén descargados"""
        models = ["qwen2.5:7b"]

        # Listar modelos locales una sola vez
        try:
            listed = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=5
            )
            local_output = listed.stdout or ""
        except Exception as e:
            logger.warning(f"No se pudo listar modelos: {e}")
            local_output = ""

        for model in models:
            base = model.split(":")[0]
            if base in local_output:
                logger.info(f"Modelo {model} ya está descargado, se omite pull")
                continue

            logger.info(f"Descargando modelo: {model}")
            try:
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    logger.info(f"Modelo {model} listo")
                else:
                    logger.warning(f"Error descargando {model}: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout descargando {model}")
            except Exception as e:
                logger.warning(f"Error con modelo {model}: {e}")
    
    def start_monitor(self):
        """Inicia el monitor RPA"""
        logger.info("Iniciando monitor RPA...")
        log_file = open(self.log_dir / "resources/logs/monitor.log", "a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "src.interface.monitor_rpa"],
            cwd=self.project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.processes.append(("monitor", process))
        logger.info("Monitor RPA iniciado")
    
    def start_streamlit(self):
        """Inicia Streamlit"""
        logger.info("Iniciando Streamlit...")
        log_file = open(self.log_dir / "resources/logs/streamlit.log", "a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
            "src/interface/streamlit_app.py",
            "--server.headless", "true"],
            cwd=self.project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.processes.append(("streamlit", process))
        logger.info("Streamlit iniciado en http://localhost:8501")
    
    def start_all(self):
        """Inicia todos los servicios"""
        logger.info("=" * 60)
        logger.info("Iniciando SREED - Sistema Resiliente de Extracción")
        logger.info("=" * 60)
        
        # 1. Iniciar Ollama
        self.start_ollama()
        
        # 2. Asegurar modelos
        self.ensure_models()
        
        # 3. Iniciar monitor
        self.start_monitor()
        
        # 4. Iniciar Streamlit
        self.start_streamlit()
        
        logger.info("=" * 60)
        logger.info("Todos los servicios están corriendo")
        logger.info("Streamlit: http://localhost:8501")
        logger.info("Presiona Ctrl+C para detener todo")
        logger.info("=" * 60)
    
    def stop_all(self):
        """Detiene todos los servicios"""
        for name, process in reversed(self.processes):
            logger.info(f"Deteniendo {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} no respondió, forzando terminación")
                process.kill()
            except Exception as e:
                logger.error(f"Error deteniendo {name}: {e}")
        
        self.processes.clear()
    
    def run(self):
        try:
            self.start_all()
            while True:
                time.sleep(2)
                for name, process in self.processes:
                    if process.poll() is not None:
                        # Leer las últimas 5 líneas del log del proceso muerto para saber por qué cayó
                        logger.error(f"Proceso {name} murió inesperadamente (Código: {process.returncode})")
                        self.stop_all()
                        sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Interrupción recibida, apagando limpiamente...")
        finally:
            self.stop_all()


if __name__ == "__main__":
    orchestrator = SREEDOrchestrator()
    orchestrator.run()