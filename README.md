# CARLA Autonomous Driving Agent

This README provides instructions for running the autonomous driving agent in the CARLA simulation environment and a brief overview of the software components.

## Table of Contents
- Setup and Execution
- Software Architecture
- Web Interface
- Troubleshooting

## Setup and Execution

### Prerequisites
- Docker installed on your system
- Access to the workstation environment

### Step-by-Step Execution

1. **Start the Docker containers**:
   - First, start the CARLA server container:
     ```bash
     ./docker_run_server.sh
     ```
   - Then, start the client container:
     ```bash
     ./docker_run_client.sh
     ```

2. **Launch the web interface server**:
   - Open a terminal to the client container 
   - Navigate to the BehaviorAgent directory:
     ```bash
     cd team_code/BehaviorAgent
     ```
   - Start the HTTP server:
     ```bash
     python server_http.py
     ```
   - This will start the server on port 9803

3. **Run the simulation**:
   - Open another terminal and connect to the client container shell
   - Navigate to the BehaviorAgent directory:
     ```bash
     cd team_code/BehaviorAgent
     ```
   - Execute the test script:
     ```bash
     ./run_test.sh
     ```

4. **View the simulation**:
   - Open a web browser
   - Navigate to the IP address specified in the configuration file, port 9803
   - Example: `http://<IP_ADDRESS>:9803`

## Software Architecture

The autonomous driving system consists of several key modules:

### BehaviorAgent ([`userCode/BehaviorAgent/carla_behavior_agent/behavior_agent.py`](userCode/BehaviorAgent/carla_behavior_agent/behavior_agent.py ))
The core autonomous driving agent that implements various driving behaviors:
- Obstacle avoidance
- Traffic rule compliance (stop signs, traffic lights)
- Junction navigation
- Vehicle and bicycle interactions
- Overtaking maneuvers

### Logger ([`userCode/BehaviorAgent/carla_behavior_agent/logger.py`](userCode/BehaviorAgent/carla_behavior_agent/logger.py ))
A unified logging system that:
- Provides multiple logging levels (DEBUG, INFO, ACTION, WARNING, ERROR, CRITICAL)
- Logs to both console/file and web interface
- Includes deduplication to prevent log spam
- Categorizes logs for easier filtering

### Overtaking Maneuver ([`userCode/BehaviorAgent/carla_behavior_agent/overtake_maneuver.py`](userCode/BehaviorAgent/carla_behavior_agent/overtake_maneuver.py ))
Specialized module for executing safe overtaking:
- Path planning for overtaking stationary or slow vehicles
- Safety checks before and during overtakes
- Dynamic speed adjustment during overtaking

### Web Interface ([`userCode/BehaviorAgent/server_http.py`](userCode/BehaviorAgent/server_http.py ))
Flask-based web server that provides visualization and control:
- Real-time RGB camera view from the vehicle
- Depth and Bird's Eye View visualization options
- Interactive control panel showing:
  - Current speed and target speed
  - Steering, throttle, and brake values
  - Real-time logs with color coding by severity

### Streamer ([`userCode/BehaviorAgent/carla_behavior_agent/utils.py`](userCode/BehaviorAgent/carla_behavior_agent/utils.py ))
Module that streams simulation data to the web interface:
- Camera feeds (RGB, depth, BEV)
- Vehicle control values
- Agent logs

## Web Interface

The web interface provides a comprehensive view of the simulation:

- Left panel: Video feed and controls
- Right panel: Speed chart and agent logs
- Real-time updates of vehicle parameters
- Color-coded log messages by severity

## Troubleshooting

- **Connection issues**: Ensure the IP address in the configuration matches your network setup
- **Slow performance**: The simulation might require adjusting quality settings (you can adjust the camera resolution in the file [`userCode/BehaviorAgent/carla_behavior_agent/config_agent_basic.json`](userCode/BehaviorAgent/carla_behavior_agent/config_agent_basic.json))

For more detailed information about the agent's behavior and configuration options, refer to the comments in the source code files.
