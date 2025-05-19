import threading
import time
import logging

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

class LogManager:
    """
    Gestisce i messaggi di log evitando duplicati basati su ID e message‐level timeout.
    """
    def __init__(self, dedup_interval: float = 5.0, max_queue: int = 100):
        self.lock = threading.Lock()
        self.message_queue = []
        self.logged_objects = {}
        self.last_clear_timestamp = time.time()
        self.valid_levels = {"DEBUG","ACTION","INFO","WARNING","ERROR"}
        # mappatura (category, level, message) → last timestamp
        self._seen_logs: dict[tuple[str,str,str], float] = {}
        self.dedup_interval = dedup_interval
        self.max_queue = max_queue

    def add_log(self, category, message, level="INFO", object_id=None, properties=None):
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

            # log per object_id (come prima)
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
                return True
            return False

    def get_logs(self):
        with self.lock:
            return list(self.message_queue)

    def reset(self):
        """Resetta il LogManager, pulendo tutti i dati."""
        with self.lock:
            self.message_queue.clear()
            self.logged_objects.clear()
            self._seen_logs.clear()  # <--- AGGIUNGI QUESTO
            self.last_clear_timestamp = time.time()
