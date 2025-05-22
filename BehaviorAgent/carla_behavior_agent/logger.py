"""
Modulo unificato per il logging che gestisce sia i log su console/file
che quelli inviati al server web, con supporto per deduplicazione e filtri.
"""

import logging
import os
import time
import threading
from logging.handlers import RotatingFileHandler

# Define Action level for logging
ACTION = 15
logging.addLevelName(ACTION, "ACTION")

def action(self, message, *args, **kwargs):
    """Metodo per loggare a livello ACTION."""
    if self.isEnabledFor(ACTION):
        self._log(ACTION, message, args, **kwargs)

# Register the action method to the logging module
logging.Logger.action = action

class DedupFilter(logging.Filter):
    """
    Filtra messaggi identici (stesso level e testo) entro un intervallo (in secondi).
    """
    def __init__(self, interval: float = 10.0):
        super().__init__()
        self.interval = interval
        self._last_seen: dict[tuple[int,str], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelno, record.getMessage())
        now = time.time()
        last = self._last_seen.get(key, 0)
        if now - last > self.interval:
            self._last_seen[key] = now
            return True
        return False

class Logger:
    """
    Classe unificata per la gestione dei log su console/file e web.
    Combina le funzionalità di Logger e LogManager in un'unica classe.
    """
    
    def __init__(self, name="behavior_agent", log_dir="./carla_behavior_agent/logs/", 
                 level=logging.DEBUG, dedup_interval=10.0, max_queue=100):
        """
        Inizializza il logger unificato con configurazione avanzata.
        
        :param name: Nome del logger
        :param log_dir: Directory dove salvare i file di log
        :param level: Livello di logging predefinito
        :param dedup_interval: Intervallo per la deduplicazione dei messaggi identici
        :param max_queue: Dimensione massima della coda di messaggi web
        """
        # Crea la directory dei log se non esiste
        os.makedirs(log_dir, exist_ok=True)
        
        # Configura il logger base di Python
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Evita la propagazione dei log
        
        # Pulisci eventuali handler esistenti
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Configura il file handler con rotazione
        file_handler = RotatingFileHandler(
            f'{log_dir}{name}.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5,
            mode="w"
        )
        file_handler.setLevel(level)
        
        # Formattatore per i log
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(self.formatter)
        file_handler.addFilter(DedupFilter(interval=dedup_interval))
        
        # Aggiungi l'handler al logger
        self.logger.addHandler(file_handler)
        
        # Configura anche un console handler per visualizzare i log sulla console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
        # Riferimento allo streamer (impostato dopo)
        self.streamer = None
        
        # Variabili per il web logging (precedentemente in LogManager)
        self.lock = threading.Lock()
        self.message_queue = []
        self.last_clear_timestamp = time.time()
        self.valid_levels = {"DEBUG", "ACTION", "INFO", "WARNING", "ERROR", "CRITICAL"}
        # mappatura (category, level, message) → last timestamp
        self._seen_logs = {}
        self.dedup_interval = dedup_interval
        self.max_queue = max_queue
        
        # Tracciamento oggetti loggati (per evitare duplicati)
        self.logged_objects = {
            "pedestrian": {},
            "bicycle": {},
            "vehicle": {},
            "obstacle": {},
            "traffic_light": False,
            "stop_sign": {},
            "overtake": {}
        }
    
    def set_streamer(self, streamer):
        """
        Imposta lo streamer per l'invio dei log al server.
        
        :param streamer: Oggetto streamer
        """
        self.streamer = streamer
        if self.streamer:
            self.reset_logs()
    
    ###########################################
    ### Metodi per log SOLO su console/file ###
    ###########################################
    
    def debug(self, message):
        """Log a livello DEBUG su console/file."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log a livello INFO su console/file."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a livello WARNING su console/file."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log a livello ERROR su console/file."""
        self.logger.error(message)
    
    def action(self, message):
        """Log a livello ACTION su console/file."""
        self.logger.action(message)
    
    def critical(self, message):
        """Log a livello CRITICAL su console/file."""
        self.logger.critical(message)
    
    def log(self, message, level="INFO"):
        """
        Metodo generico per log su console/file.
        
        :param message: Messaggio da loggare
        :param level: Livello di log
        """
        level = level.upper()
        
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "ACTION":
            self.logger.action(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
    
    ########################################
    ### Metodi per log SOLO sul web ###
    ########################################
    
    def web_debug(self, category, message, object_id=None, properties=None):
        """Log a livello DEBUG solo sul web."""
        self.add_log(category, message, "DEBUG", object_id, properties)
    
    def web_info(self, category, message, object_id=None, properties=None):
        """Log a livello INFO solo sul web."""
        self.add_log(category, message, "INFO", object_id, properties)
    
    def web_warning(self, category, message, object_id=None, properties=None):
        """Log a livello WARNING solo sul web."""
        self.add_log(category, message, "WARNING", object_id, properties)
    
    def web_error(self, category, message, object_id=None, properties=None):
        """Log a livello ERROR solo sul web."""
        self.add_log(category, message, "ERROR", object_id, properties)
    
    def web_action(self, category, message, object_id=None, properties=None):
        """Log a livello ACTION solo sul web."""
        self.add_log(category, message, "ACTION", object_id, properties)
    
    def web_critical(self, category, message, object_id=None, properties=None):
        """Log a livello CRITICAL solo sul web."""
        self.add_log(category, message, "CRITICAL", object_id, properties)
    
    def web_log(self, category, message, level="INFO", object_id=None, properties=None):
        """
        Metodo generico per log solo sul web.
        
        :param category: Categoria del log
        :param message: Messaggio da loggare
        :param level: Livello di log
        :param object_id: ID dell'oggetto per tracciamento duplicati
        :param properties: Proprietà aggiuntive
        """
        self.add_log(category, message, level, object_id, properties)
    
    ##################################################
    ### Metodi per log SIA su console/file CHE web ###
    ##################################################
    
    def both_debug(self, category, message, object_id=None, properties=None):
        """Log a livello DEBUG sia su console/file che sul web."""
        self.debug(f"[{category}] {message}")
        self.add_log(category, message, "DEBUG", object_id, properties)
    
    def both_info(self, category, message, object_id=None, properties=None):
        """Log a livello INFO sia su console/file che sul web."""
        self.info(f"[{category}] {message}")
        self.add_log(category, message, "INFO", object_id, properties)
    
    def both_warning(self, category, message, object_id=None, properties=None):
        """Log a livello WARNING sia su console/file che sul web."""
        self.warning(f"[{category}] {message}")
        self.add_log(category, message, "WARNING", object_id, properties)
    
    def both_error(self, category, message, object_id=None, properties=None):
        """Log a livello ERROR sia su console/file che sul web."""
        self.error(f"[{category}] {message}")
        self.add_log(category, message, "ERROR", object_id, properties)
    
    def both_action(self, category, message, object_id=None, properties=None):
        """Log a livello ACTION sia su console/file che sul web."""
        self.action(f"[{category}] {message}")
        self.add_log(category, message, "ACTION", object_id, properties)
    
    def both_critical(self, category, message, object_id=None, properties=None):
        """Log a livello CRITICAL sia su console/file che sul web."""
        self.critical(f"[{category}] {message}")
        self.add_log(category, message, "CRITICAL", object_id, properties)
    
    def both_log(self, category, message, level="INFO", object_id=None, properties=None):
        """
        Metodo generico per log sia su console/file che sul web.
        
        :param category: Categoria del log
        :param message: Messaggio da loggare
        :param level: Livello di log
        :param object_id: ID dell'oggetto per tracciamento duplicati
        :param properties: Proprietà aggiuntive
        """
        self.log(f"[{category}] {message}", level)
        self.add_log(category, message, level, object_id, properties)
    
    ##################################################
    ### Metodi ex-LogManager incorporati ###
    ##################################################
    
    def add_log(self, category, message, level="INFO", object_id=None, properties=None):
        """
        Aggiunge un log alla coda web (precedentemente in LogManager).
        
        :param category: Categoria del log
        :param message: Messaggio da loggare
        :param level: Livello di log
        :param object_id: ID dell'oggetto per tracciamento duplicati
        :param properties: Proprietà aggiuntive
        :return: True se il log è stato aggiunto, False altrimenti
        """
        with self.lock:
            level = level.upper()
            if level not in self.valid_levels:
                level = "INFO"

            now = time.time()
            key = (category, level, message)
            last = self._seen_logs.get(key, 0)
            # scarta se è troppo recente
            if now - last < self.dedup_interval:
                return False
            self._seen_logs[key] = now

            # log per object_id
            if object_id is not None:
                self.logged_objects.setdefault(category, {})
                if object_id in self.logged_objects[category]:
                    return False
                self.logged_objects[category][object_id] = {
                    'timestamp': now,
                    'properties': properties or {}
                }
                if "ID:" not in message and "id:" not in message.lower():
                    message = f"{message} (ID: {object_id})"

            entry = {
                "category": category,
                "message": message,
                "level": level,
                "timestamp": now,
                "formatted_time": time.strftime("%H:%M:%S", time.localtime(now)),
                "log_id": f"{now}-{category}-{object_id or 'none'}"
            }
            
            # rispetta clear_timestamp e lunghezza coda
            if now > self.last_clear_timestamp:
                self.message_queue.append(entry)
                if len(self.message_queue) > self.max_queue:
                    self.message_queue.pop(0)
                
                # Invia il log allo streamer se disponibile
                if self.streamer:
                    self.streamer.add_log(category, message, level, object_id, properties)
                
                return True
            return False

    def get_logs(self):
        """
        Restituisce la coda dei messaggi web.
        
        :return: Lista dei messaggi nella coda
        """
        with self.lock:
            return list(self.message_queue)

    def reset_logs(self):
        """Resetta il sistema di logging web, pulendo tutti i dati."""
        with self.lock:
            self.message_queue.clear()
            self.logged_objects = {
                "pedestrian": {},
                "bicycle": {},
                "vehicle": {},
                "obstacle": {},
                "traffic_light": False,
                "stop_sign": {},
                "overtake": {}
            }
            self._seen_logs.clear()
            self.last_clear_timestamp = time.time()
            
            # Invia un messaggio di reset allo streamer se disponibile
            if self.streamer:
                self.streamer.reset_logs()