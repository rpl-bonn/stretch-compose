# Stretch Robot Setup

This file describes how to [setup a connection](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/#untethered-setup) between the robot and the workstation via Moonlight (Theoretically, SSH is also an option since only Terminal commands are used, but then the FUNMAP mapping is not visualized). It also describes the steps necessary everytime [after turning on the robot](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/#robot-overview) before launching anything else and how to properly shut down the robot.


## Untethered Robot Connection
1. The workstation needs to be connected to the **same network** as the robot
2. Setup a **wireless connection** between workstation & robot 
	1. Install **[Moonlight](https://moonlight-stream.org/)** on the workstation 
	2. Get access to the **robot desktop** (either through HDMI+keyboard+mouse OR someone else who has already a connection via Moonlight)
	3. Click on the detected Stretch connection in Moonlight to get a **4 digit PIN**
	4. Open **Sunshine** on the robot (orange icon at the top right), navigate to the PIN tab & enter the 4 digit PIN
	5. If HDMI was used, replace HDMI cable with **HDMI dongle**
	6. Optimize **streaming settings** (4K Resolution, 60 FPS, 3-5 Bitrate, choose preferred display mode, select "Optimize mouse for remote desktop instead of games")
3. Now you can **start the robot desktop stream** in Moonlight
	1. Only **one person at a time** can connect via Moonlight 
	2. To **exit the stream** press: `Ctrl + Alt + Shift + Q`


## Starting the Robot
1. Run `stretch_free_robot_process.py` to **free** the **robot process** from gamepad teleoperation
2. Run `stretch_robot_home.py` to **home** the robot **joints**
3. Optional: Run `stretch_system_check.py` to **check** the robot **status**


## Shutting down the Robot
1. Run `stretch_gamepad_teleop.py` to connect the **controller** with the robot
2. Press the **Back-Button** on the controller for a few seconds to shut down
	1. This will put the robot into a stow position
3. Wait until the **LiDar stops spinning**, then turn off the robot at the body switch


