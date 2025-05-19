import cv2
from flask import Flask, Response, request, jsonify

import numpy as np
import threading
import base64
import matplotlib.pyplot as plt
from flask import send_file
from io import BytesIO
import time
from collections import deque


app = Flask(__name__)

class ServerData:
    def __init__(self) -> None:
        self.__rgb_image = np.zeros((360,640,3), np.uint8)
        self.__depth_image = np.zeros((360,640,3), np.uint8)
        self.__bev_image = np.zeros((360,640,3), np.uint8)

        self.__controls = {"throttle" : 0, "steer" : 0,"brake" : 0}
        
        # Aggiungi una struttura per i log
        self.__logs = deque(maxlen=50)  # Mantiene solo gli ultimi 50 log

        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.bev_lock = threading.Lock()
        self.controls_lock = threading.Lock()
        self.logs_lock = threading.Lock()
        
        self.__record_file = "Speed.txt"
        self.__last_clear_timestamp = time.time()  # Timestamp per l'ultimo clear
        
    def __record_data(self, data):
        if self.__record_file is not None:
            with open(self.__record_file, "a") as fp:
                fp.write(str(data)+"\n")
                fp.close()
                
    def setRGBImage(self,rgb_image):
        self.rgb_lock.acquire()
        self.__rgb_image = rgb_image
        self.rgb_lock.release()
    
    def getRGBImage(self):
        return self.__rgb_image

    def setDepthImage(self,depth_image):
        self.depth_lock.acquire()
        self.__depth_image = depth_image
        self.depth_lock.release()
    
    def getDepthImage(self):
        return self.__depth_image

    def setBEVImage(self,bev_image):
        self.bev_lock.acquire()
        self.__bev_image = bev_image
        self.bev_lock.release()
    
    def getBEVImage(self):
        return self.__bev_image

    def setControls(self, controls):
        self.controls_lock.acquire()
        self.__controls = {
            "throttle": controls.get("throttle", 0),
            "steer": controls.get("steer", 0),
            "brake": controls.get("brake", 0),
            "target_speed": controls.get("target_speed", 0),
            "current_speed": controls.get("current_speed", 0)
        }
        self.controls_lock.release()
    
    def getControls(self):
        return self.__controls
        
    def setLogs(self, logs):
        """Aggiunge nuovi log alla coda, ignorando quelli più vecchi del clear"""
        self.logs_lock.acquire()
        for log in logs:
            # Aggiungi solo se il log è più recente dell'ultimo clear
            if log["timestamp"] > self.__last_clear_timestamp:
                log["formatted_time"] = time.strftime("%H:%M:%S", time.localtime(log["timestamp"]))
                self.__logs.append(log)
        self.logs_lock.release()
        
    def getLogs(self):
        """Restituisce i log correnti"""
        return list(self.__logs)
    
    def clearLogs(self):
        """Pulisce tutti i log e aggiorna il timestamp di clear"""
        self.logs_lock.acquire()
        self.__logs.clear()
        self.__last_clear_timestamp = time.time()
        self.logs_lock.release()

def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args)
        thread.setDaemon(True)
        thread.start()
        return thread
    return wrapper
                    
appData = ServerData()

def sendImagesToWeb(getFrame, lock):
    while True:
        try:
            lock.acquire()
            frame = getFrame()
            jpg = cv2.imencode('.jpg', frame)[1]
            lock.release()
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+bytearray(jpg)+b'\r\n'
        except Exception as e:
            print("Something went wrong: ", e)
            if lock.locked():
                lock.release()
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+bytearray()+b'\r\n'

@app.route('/rgbimage')
def rgbimage():
    global appData
    return Response(sendImagesToWeb(appData.getRGBImage, appData.rgb_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depthimage')
def depthimage():
    global appData
    return Response(sendImagesToWeb(appData.getDepthImage, appData.depth_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bevimage')
def bevimage():
    global appData
    return Response(sendImagesToWeb(appData.getBEVImage, appData.bev_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/controls', methods=['POST'])
def controls():
    global appData
    appData.controls_lock.acquire()
    controls = appData.getControls()
    appData.controls_lock.release()
    return jsonify(controls)

@app.route('/logs', methods=['POST'])
def logs():
    """Restituisce i log correnti"""
    global appData
    appData.logs_lock.acquire()
    logs = appData.getLogs()
    appData.logs_lock.release()
    print(f"Sending logs to UI: {len(logs)} entries")
    return jsonify(logs)

@app.route('/new_frame', methods=['POST']) 
def new_frame():
    global appData
    data = request.json

    if data is None or "type" not in data:
        return jsonify(success=False, error="No data or missing 'type' field"), 400

    if data["type"] == "RGB":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setRGBImage(dataArray)

    elif data["type"] == "Depth":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setDepthImage(dataArray)
    
    elif data["type"] == "Controls":
        appData.setControls(data["data"])
    
    elif data["type"] == "BEV":
        dataType = np.dtype(data["data"]["dtype"])
        dataArray = np.frombuffer(base64.b64decode(data["data"]["encode"].encode('utf-8')), dataType)
        dataArray = dataArray.reshape(data["data"]["shape"])  
            
        appData.setBEVImage(dataArray)
    
    elif data["type"] == "Logs":
        # Gestione dei log ricevuti
        appData.setLogs(data["data"]["logs"])

    return jsonify(success=True)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    appData.clearLogs()  # <--- CORRETTO
    return jsonify({"status": "cleared", "timestamp": time.time()})


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Carla Simulator </title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <style>
            /* Stili esistenti */
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
                padding-top: 0;
                padding-bottom: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .main-container {
                display: flex;
                justify-content: center;
                gap: 20px;
                flex: 1;
                width: 100%;
            }
            
            #left-panel, #right-panel {
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 0 12px rgba(0,0,0,0.1);
            }
            
            #left-panel {
                flex: 2 1 60%;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }
            
            #right-panel {
                flex: 1 1 30%;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            #video-container {
                width: 100%;
                overflow: hidden;
                border-radius: 8px;
                border: 1px solid #e1e1e1;
            }
            
            #video-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            
            /* Stili migliorati per i controlli */
            #controls {
                width: 100%;
                background: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 0 8px rgba(0,0,0,0.05);
                border: 1px solid #e1e1e1;
                padding-top: 10px;
                padding-bottom: 10px;
                display: flex;
                justify-content: space-between;
            }
            
            .control-item {
                flex: 1;
                padding: 0 10px;
                text-align: center;
                border-right: 1px solid #e1e1e1;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            
            .control-item:last-child {
                border-right: none;
            }
            
            .control-label {
                font-size: 13px;
                color: #666;
                text-transform: uppercase;
                margin-bottom: 5px;
                font-weight: bold;
            }
            
            .control-value {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            
            .control-bar {
                width: 100%;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                overflow: hidden;
                margin-top: 3px;
            }
            
            .control-bar-fill {
                height: 100%;
                transition: width 0.3s ease, background-color 0.3s;
            }
            
            .throttle-fill { background-color: #28a745; }
            .brake-fill { background-color: #dc3545; }
            .steer-fill { background-color: #007bff; }
            
            #chart-container {
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 0 8px rgba(0,0,0,0.1);
                padding: 15px;
                border: 1px solid #e1e1e1;
            }
            
            #logs-container {
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 0 8px rgba(0,0,0,0.1);
                padding: 15px;
                border: 1px solid #e1e1e1;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            canvas {
                width: 100% !important;
                height: 300px !important;
            }
            
            .logs-content {
                overflow-y: auto;
                max-height: 400px;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                font-family: monospace;
                font-size: 13px;
            }
            
            .log-entry {
                margin: 5px 0;
                padding: 8px;
                border-radius: 4px;
                border-left: 4px solid;
            }
            
            .log-INFO { 
                background-color: #d1ecf1; 
                color: #0c5460; 
                border-left-color: #0c5460;
            }
            
            .log-DEBUG { 
                background-color: #e2e3e5; 
                color: #383d41; 
                border-left-color: #383d41;
            }
            
            .log-WARNING { 
                background-color: #fff3cd; 
                color: #856404; 
                border-left-color: #856404;
            }
            
            .log-ERROR { 
                background-color: #f8d7da; 
                color: #721c24; 
                border-left-color: #721c24;
            }
            
            
            .log-ACTION { 
                background-color: #ffe8d6; 
                color: #a05500; 
                border-left-color: #ff8c00;
            }
            
            .log-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .legend-color {
                width: 15px;
                height: 15px;
                border-radius: 3px;
            }
            
            h2 {
                margin: 0;
                padding: 0;
                color: #333;
            }
            
            footer {
                text-align: center;
                width: 100%;
                color: #888;
                margin-top: 20px;
                padding: 10px 0;
            }
            
            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                width: 100%;
            }
            
            .refresh-button {
                background: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
            }
            
            .refresh-button:hover {
                background: #0069d9;
            }
            
            #speedometer {
                margin-top: -150px;
                font-size: 15px; 
                font-weight: bold; 
                color: #007BFF; 
                text-align: center;
            }

        </style>
    </head>
    <body>
        <div class="main-container">
            <div id="left-panel">
                <h1>Carla Simulator</h1>
                <div id="video-container">
                    <img src="/rgbimage" />
                </div>
                <div id="controls">Loading...</div>
                <div id="speedometer">0 km/h</div>
            </div>
            <div id="right-panel">
                <div id="chart-container">
                    <div class="panel-header">
                        <h2>Speed Chart</h2>
                    </div>
                    <canvas id="speedChart"></canvas>
                </div>
                <div id="logs-container">
                    <div class="panel-header">
                        <h2>Agent Logs</h2>
                        <button class="refresh-button" onclick="clearLogs()">Clear</button>
                    </div>
                    <div class="log-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #d1ecf1; border: 1px solid #0c5460;"></div>
                            <span>Info</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #e2e3e5; border: 1px solid #383d41;"></div>
                            <span>Debug</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #ffe8d6; border: 1px solid #a05500;"></div>
                            <span>Action</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #fff3cd; border: 1px solid #856404;"></div>
                            <span>Warning</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #f8d7da; border: 1px solid #721c24;"></div>
                            <span>Error</span>
                        </div>
                        
                    </div>
                    <div class="logs-content">
                        <div class="log-entry">Waiting for logs...</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let targetData = [], currentData = [], labels = [];
            const ctx = document.getElementById('speedChart').getContext('2d');
            const speedChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Target Speed', data: targetData, borderColor: 'rgba(255,99,132,1)', fill: false },
                        { label: 'Current Speed', data: currentData, borderColor: 'rgba(54,162,235,1)', fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    animation: { duration: 0 },
                    maintainAspectRatio: false,
                    scales: { 
                        x: { display: false },
                        y: { 
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Speed (km/h)'
                            }
                        } 
                    }
                }
            });

            function getControls() {
                $.ajax({
                    type: 'POST', url: '/controls', dataType: 'json',
                    success: function(result) {
                        // Visualizzazione migliorata dei controlli
                        let controlsHTML = `
                            <div class="control-item">
                                <div class="control-label">Throttle</div>
                                <div class="control-value">${result.throttle.toFixed(2)}</div>
                                <div class="control-bar">
                                    <div class="control-bar-fill throttle-fill" style="width: ${result.throttle * 100}%"></div>
                                </div>
                            </div>
                            <div class="control-item">
                                <div class="control-label">Steer</div>
                                <div class="control-value">${result.steer.toFixed(2)}</div>
                                <div class="control-bar">
                                    <div class="control-bar-fill steer-fill" style="width: ${((result.steer + 1) / 2) * 100}%"></div>
                                </div>
                            </div>
                            <div class="control-item">
                                <div class="control-label">Brake</div>
                                <div class="control-value">${result.brake.toFixed(2)}</div>
                                <div class="control-bar">
                                    <div class="control-bar-fill brake-fill" style="width: ${result.brake * 100}%"></div>
                                </div>
                            </div>
                        `;
                        
                        document.getElementById('controls').innerHTML = controlsHTML;
                        document.getElementById('speedometer').innerHTML = Math.round(result.current_speed) + " km/h";

                        const ts = new Date().toLocaleTimeString();
                        labels.push(ts); targetData.push(result.target_speed); currentData.push(result.current_speed);
                        if (labels.length > 30) { labels.shift(); targetData.shift(); currentData.shift(); }
                        speedChart.update();
                    },
                    complete: function() { setTimeout(getControls, 500); }
                });
            }
            
            // Variabile per memorizzare l'ultimo timeout
            let logsTimeout;

            function getLogs() {
                $.ajax({
                    type: 'POST', url: '/logs', dataType: 'json',
                    success: function(logs) {
                        if (logs && logs.length > 0) {
                            const logsContainer = document.querySelector('#logs-container .logs-content');
                            
                            // Svuota il contenitore solo se ci sono nuovi log
                            logsContainer.innerHTML = '';
                            
                            // Mantieni un set di ID log già visualizzati
                            const loggedIds = new Set();
                            
                            for (let i = logs.length - 1; i >= 0; i--) {
                                const log = logs[i];
                                
                                // Verifica se il log ha già un ID univoco, altrimenti ne crea uno
                                const logId = log.log_id || `${log.timestamp}-${log.category}`;
                                
                                // Controlla se questo log è già stato visualizzato
                                if (!loggedIds.has(logId)) {
                                    loggedIds.add(logId);
                                    
                                    const logEntry = document.createElement('div');
                                    logEntry.className = `log-entry log-${log.level}`;
                                    logEntry.innerHTML = `<strong>${log.formatted_time} [${log.category}]</strong>: ${log.message}`;
                                    logsContainer.appendChild(logEntry);
                                }
                            }
                        }
                    },
                    complete: function() { 
                        logsTimeout = setTimeout(getLogs, 1000); 
                    }
                });
            }
            
            function clearLogs() {
                $.ajax({
                    type: 'POST',
                    url: '/clear_logs',
                    success: function() {
                        const logsContainer = document.querySelector('#logs-container .logs-content');
                        logsContainer.innerHTML = '<div class="log-entry">Logs cleared</div>';
                        
                        // Cancella il timeout corrente e crea un ritardo più lungo prima di richiedere nuovi log
                        clearTimeout(logsTimeout);
                        logsTimeout = setTimeout(getLogs, 2000);
                    }
                });
            }
            
            getControls();
            getLogs();
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9803, debug=False)