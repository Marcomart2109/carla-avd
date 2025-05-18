import socket
import numpy as np
import requests
import threading 
import time
import base64
import matplotlib
from matplotlib import pyplot as plt

def threaded(func):
    def wrapper(*k, **kw):
        thread = threading.Thread(target=func,args=k,kwargs=kw, daemon=True)
        thread.start()
        return thread
    return wrapper

class Plot():
    def __init__(self, filename, plotname) -> None:
        self.filename = filename
        self.plotname = plotname
        
    def plot(self):
        x_data = []
        
        cur_data = []
        target_data = []
        
        with open(self.filename, "r") as fp:
            for line in fp:
                timestamp, speed, target = line.split(";")
                x_data.append(float(timestamp))
                cur_data.append(float(speed))
                target_data.append(float(target))
            fp.close()
        plt.plot(x_data, cur_data)
        plt.plot(x_data, target_data)
        plt.savefig(self.plotname)
        
        
        
class Streamer():

    def __init__(self, IP):
        self.run = True
        self.verbose = False

        # Inizializza il LogManager
        self.log_manager = LogManager(min_interval=1.0)

        self.data = {
                "url": "http://"+IP+":9803/new_frame",
                "RGB": {
                    "frame_lock": threading.Lock(),
                    "frame": None,
                    "update": False
                },
                "Depth": {
                    "frame_lock": threading.Lock(),
                    "frame": None,
                    "update": False
                },
                "BEV": {
                    "frame_lock": threading.Lock(),
                    "frame": None,
                    "update": False
                },
                "Controls": {
                    "data_lock": threading.Lock(),
                    "data": None,
                    "update": False
                },
                "Logs": {
                    "data_lock": threading.Lock(),
                    "data": None,
                    "update": False
                }
        }

        # launch the streamer threads
        threading.Thread(target=self.sendRGBImage, name="RGBStreamer").start()
        threading.Thread(target=self.sendDepthImage, name="DepthStreamer").start()
        threading.Thread(target=self.sendBEVImage, name="BEVStreamer").start()
        threading.Thread(target=self.sendControlsData, name="ControlsDataStreamer").start()
        threading.Thread(target=self.sendLogsData, name="LogsDataStreamer").start()
        

    def sendRGBImage(self):
        self.__sendImage("RGB")

    def sendDepthImage(self):   
        self.__sendImage("Depth")
    
    def sendBEVImage(self):   
        self.__sendImage("BEV")
    
    def sendControlsData(self):   
        self.__sendData("Controls")
    
    def sendLogsData(self):
        """Thread che invia periodicamente i log accumulati"""
        self.__sendData("Logs")
    
    def __sendData(self, datatype):
        while self.run:
            try:
                if self.data[datatype]["data"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["data_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : self.data[datatype]["data"]
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["data_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["data_lock"].locked():
                    self.data[datatype]["data_lock"].release() 

    def __sendImage(self, datatype):
        ''''
        Take the objects image and send it
        '''
        while self.run:
            try:
                if self.data[datatype]["frame"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["frame_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : {
                            "encode" : base64.b64encode(self.data[datatype]["frame"].tobytes()).decode('utf-8'),
                            "dtype" : str(self.data[datatype]["frame"].dtype),
                            "shape" : self.data[datatype]["frame"].shape
                        }
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["frame_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["frame_lock"].locked():
                    self.data[datatype]["frame_lock"].release()
        print("---- Stream finished ----")
    
    def send_data(self, datatype, data):
        self.data[datatype]["data_lock"].acquire()
        self.data[datatype]["data"] = data
        self.data[datatype]["update"] = True
        self.data[datatype]["data_lock"].release()

    def send_frame(self, datatype, frame):
        self.data[datatype]["frame_lock"].acquire()
        self.data[datatype]["frame"] = frame
        self.data[datatype]["update"] = True
        self.data[datatype]["frame_lock"].release()

    def add_log(self, category, message, level="INFO"):
        """
        Aggiunge un log al LogManager e lo prepara per l'invio.
        """
        if self.log_manager.add_log(category, message, level):
            # Ottiene tutti i log e li prepara per l'invio
            logs = self.log_manager.get_logs()
            self.send_data("Logs", {"logs": logs})

    def reset_logs(self):
        """Resetta i log e invia l'aggiornamento al server"""
        self.log_manager.reset()
        self.send_data("Logs", {"logs": []})

class LogManager:
    """
    Gestisce i messaggi di log con throttling temporale per evitare duplicati ravvicinati.
    """
    def __init__(self, min_interval=1.0):
        """
        Inizializza il LogManager.
        
        :param min_interval: Intervallo minimo in secondi tra due eventi dello stesso tipo
        """
        # Dizionario che memorizza {categoria_evento: {messaggio: ultimo_timestamp}}
        self.last_events = {}
        # Intervallo minimo tra due eventi identici (in secondi)
        self.min_interval = min_interval
        # Coda dei messaggi da inviare
        self.message_queue = []
        # Lock per gestire l'accesso concorrente
        self.lock = threading.Lock()
    
    def add_log(self, category, message, level="INFO"):
        """
        Aggiunge un log se non è un duplicato recente.
        """
        with self.lock:
            current_time = time.time()
            
            # Inizializza il dizionario per questa categoria se non esiste
            if category not in self.last_events:
                self.last_events[category] = {}
            
            # Verifica se il messaggio è già stato inviato recentemente
            if (message not in self.last_events[category] or 
                current_time - self.last_events[category][message] > self.min_interval):
                
                # Aggiungi il messaggio alla coda
                self.message_queue.append({
                    "category": category,
                    "message": message,
                    "level": level,
                    "timestamp": current_time
                })
                # Aggiorna il timestamp dell'ultimo evento
                self.last_events[category][message] = current_time
                
                # Limita la dimensione della coda (opzionale)
                if len(self.message_queue) > 100:
                    self.message_queue.pop(0)
                
                return True
            return False
    
    def get_logs(self):
        """
        Restituisce tutti i messaggi in coda e li mantiene.
        """
        with self.lock:
            return self.message_queue.copy()
    
    def clear_old_logs(self, max_age=30.0):
        """
        Rimuove i log più vecchi di max_age secondi.
        """
        with self.lock:
            current_time = time.time()
            self.message_queue = [log for log in self.message_queue 
                                if current_time - log["timestamp"] < max_age]

    def reset(self):
        """Resetta il log manager, pulendo tutti i log e gli eventi registrati"""
        with self.lock:
            self.last_events = {}
            self.message_queue = []


if __name__ == "__main__":
    speed_plot = Plot("./userCode/speed.txt", "./userCode/speedplot.png")
    speed_plot.plot()