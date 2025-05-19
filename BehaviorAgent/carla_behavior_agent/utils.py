import requests
import threading 
import time
import base64
from matplotlib import pyplot as plt
from log_manager import LogManager 

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
        self.log_manager = LogManager()

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

    def add_log(self, category, message, level="INFO", object_id=None, properties=None):
        """
        Aggiunge un log al LogManager e lo prepara per l'invio.
        """
        if self.log_manager.add_log(category, message, level, object_id, properties):
            # Ottiene tutti i log e li prepara per l'invio
            logs = self.log_manager.get_logs()
            self.send_data("Logs", {"logs": logs})

    def reset_logs(self):
        """Resetta i log e invia l'aggiornamento al server"""
        self.log_manager.reset()
        self.send_data("Logs", {"logs": []})