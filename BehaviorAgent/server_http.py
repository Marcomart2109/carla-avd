import cv2
from flask import Flask, Response, request, jsonify

import numpy as np
import threading
import base64
import matplotlib.pyplot as plt
from flask import send_file
from io import BytesIO


app = Flask(__name__)

class ServerData:
    def __init__(self) -> None:
        self.__rgb_image = np.zeros((360,640,3), np.uint8)
        self.__depth_image = np.zeros((360,640,3), np.uint8)
        self.__bev_image = np.zeros((360,640,3), np.uint8)

        self.__controls = {"throttle" : 0, "steer" : 0,"brake" : 0}

        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.bev_lock = threading.Lock()
        self.controls_lock = threading.Lock()
        
        self.__record_file = "Speed.txt"
        
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

@app.route('/new_frame', methods=['POST']) 
def new_frame():
    global appData
    data = request.json

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

    return jsonify(success=True)

def ajaxClient():
    javascript_code = "<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js\"></script>"\
        "<script type=\"text/javascript\">"\
        "function getControls(){"\
            "$.ajax({"\
                "type: 'POST',"\
                "url: '/controls',"\
                "contentType: 'application/json',"\
                "dataType: 'json',"\
                "success: function(result) {"\
                    "document.getElementById('controls').innerHTML = " \
                        "\"Throttle: \" + result.throttle.toFixed(2) + " \
                        "\" | Steer: \" + result.steer.toFixed(2) + " \
                        "\" | Brake: \" + result.brake.toFixed(2) + " \
                        "\" | Target Speed: \" + result.target_speed.toFixed(2) + " \
                        "\" | Current Speed: \" + result.current_speed.toFixed(2);" \
                "},"\
                "complete: function(){"\
                    "setTimeout(getControls, 1000);"\
                "}"\
            "});"\
        "};"\
        "getControls();"\
    "</script>"

    return javascript_code

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Carla Simulator</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                gap: 20px;
                height: 100vh;
            }
            #left-panel, #right-panel {
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 0 12px rgba(0,0,0,0.1);
                padding: 20px;
            }
            #left-panel {
                flex: 2 1 60%;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
            }
            #video-container {
                width: 100%;
                overflow: hidden;
                border-radius: 8px;
            }
            #video-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            #controls {
                font-size: 18px;
                color: #444;
                background: #fafafa;
                padding: 10px;
                border-radius: 8px;
                width: 100%;
                text-align: center;
                box-shadow: 0 0 8px rgba(0,0,0,0.05);
            }
            #right-panel {
                flex: 1 1 30%;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            #chart-container {
                flex: 1;
            }
            canvas {
                width: 100% !important;
                height: 200px !important;
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 0 8px rgba(0,0,0,0.1);
            }
            footer {
                position: absolute;
                bottom: 10px;
                text-align: center;
                width: 100%;
                color: #888;
            }
        </style>
    </head>
    <body>
        <div id="left-panel">
            <h1>Carla Simulator</h1>
            <div id="video-container">
                <img src="/rgbimage" />
            </div>
            <div id="controls">Loading...</div>
            <div id="speedometer" style="font-size: 48px; font-weight: bold; color: #007BFF; text-align: center;">0 km/h</div>
        </div>
        <div id="right-panel">
            <h2>Speed Chart</h2>
            <div id="chart-container">
                <canvas id="speedChart"></canvas>
            </div>
        </div>
        <footer>SimpleVisualizer</footer>

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
                    scales: { x: { display: false }, y: { beginAtZero: true } }
                }
            });

            function getControls() {
                $.ajax({
                    type: 'POST', url: '/controls', dataType: 'json',
                    success: function(result) {
                        document.getElementById('controls').innerHTML =
                            `Throttle: ${result.throttle.toFixed(2)} | Steer: ${result.steer.toFixed(2)} | Brake: ${result.brake.toFixed(2)}`;
                            
                        document.getElementById('speedometer').innerHTML = Math.round(result.current_speed) + " km/h";

                        const ts = new Date().toLocaleTimeString();
                        labels.push(ts); targetData.push(result.target_speed); currentData.push(result.current_speed);
                        if (labels.length > 30) { labels.shift(); targetData.shift(); currentData.shift(); }
                        speedChart.update();
                    },
                    complete: function() { setTimeout(getControls, 500); }
                });
            }
            getControls();
        </script>
    </body>
    </html>
    """



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9803, debug=True)