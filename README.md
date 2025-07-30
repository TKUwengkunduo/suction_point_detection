# Suction Point Detection

The main function of this repository is to identify the suction cup suction point.

---

## Installation

### Build a workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### Install this repository 
```bash
git clone https://github.com/TKUwengkunduo/suction_point_detection.git
cd ~/ros2_ws/
```


---

## Use
### 1. Collecting information
You will get a `hand_eye_data/data.json` file and some photos.
```bash
colcon build
source install/setup.bash
ros2 run suction_point_detection ransac
```


### 2. Simulation
```bash
ros2 topic pub /spd/plane_roi vision_msgs/msg/BoundingBox2D "{center: {position: {x: 400.0, y: 400.0}, theta: 0.0}, size_x: 100.0, size_y: 100.0}" --rate 10
```