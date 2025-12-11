# WildScenes ROS2 Segmentation
Real-time segmentation using WildScenes Cylinder3D model in ROS2 environment. LiDAR (Livox MID-360) point cloud → WildScenes → Segmented Cloud with accurate color representation.

## Architecture

- **LiDAR Data**  
  - Streams raw point clouds from the Livox MID-360 sensor.
- **ROS2 Node (`veggie_drive_pkg`)**  
  - Subscribes to `/livox/lidar`, preprocesses points, runs inference, and publishes `/wildscenes_segmented`.
- **WildScenes Cylinder3D Model**  
  - Performs semantic segmentation on the filtered point cloud and returns class labels.

## Files
- `veggie_ws/src/veggie_drive_pkg/` - ROS2 package with segmentation node
- `wildscenes/` - Model configuration and utilities
- `Dockerfile.unified` - Docker environment for x86
- 'Dockerfile.jet.foxy' - Docker environment for jetson
- `segmented_points.rviz` - RViz configuration for visualization

## Quick Start

### 1. Prerequisites
- **Laptop with GPU** or **Nvidia Jetson**
- **Docker**

### 2.1 Build and Run X86

```bash
# Build Docker image
docker build -f Dockerfile.unified -t veggie-drive:dev .

# Run container with GPU support and X11 forwarding
docker run --rm -it --network host --gpus all \
  --user $(id -u):$(id -g) \
  -e DISPLAY -e XAUTHORITY -e QT_X11_NO_MITSHM=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e USER=$USER -e HOME=$HOME \
  -e XDG_RUNTIME_DIR=/tmp/runtime-$(id -u) \
  -e ROS_LOG_DIR=$HOME/.ros/log \
  -e ROS_DISTRO=humble \
  -v $XAUTHORITY:$XAUTHORITY:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/dri:/dev/dri \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v $(pwd):/veggie_drive \
  veggie-drive:dev bash

# Inside container: Build ROS2 workspace
cd /veggie_drive/veggie_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select veggie_drive_pkg
source install/setup.bash
```

### 2.2 Build and Run Jetson
```bash
# docker build -f Dockerfile.jetson.wildscenes -t wild:fox .
docker build -f Dockerfile.jetson.foxy -t foxy:jetsont1 .
# docker build --no-cache -f Dockerfile.jetson.foxy -t foxy:jetsont1 .

# version with rviz2 running on host
docker run --rm -it \
 --network host \
 --runtime=nvidia \
 -e NVIDIA_DRIVER_CAPABILITIES=all \
 -e ROS_DOMAIN_ID=42 \
 -e ROS_LOCALHOST_ONLY=0 \
 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
 -v $(pwd):/veggie_drive \
 foxy:jetsont1 \
 bash

cd /veggie_drive/veggie_ws
source /opt/ros/foxy/install/setup.bash
rm -rf build install log
colcon build --packages-select veggie_drive_pkg
source install/setup.bash
```

### 2.3 Circular import fix
```bash
pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
sudo apt-get remove -y opencv-dev opencv-libs
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  libopencv-dev python3-opencv \
  libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
export OPENCV_DISABLE_OPENCL=1
export OPENCV_IO_ENABLE_OPENEXR=0
```

### 2.4 Tmux
```bash
tmux
tmux set -g mouse on
tmux split-window -h
tmux split-window -v
```

### 3. Launch Segmentation System

```bash
# Terminal 1: Launch segmentation node
ros2 launch veggie_drive_pkg wildscenes_launch.py

# Terminal 2: Play LiDAR data (if you have bag files)
ros2 bag play data/livox_data_jetson_0.db3 --rate 1.0

# Launch RViz for visualization on host machine
source /opt/ros/foxy/setup.bash
ros2 run rviz2 rviz2 --display-config /home/unitree/Desktop/veggie_drive/segmented_points.rviz
```

### 4. Monitor Results

```bash
# on the Jetson (host not docker)
source /opt/ros/foxy/setup.bash
rviz2

# Check topics
ros2 topic list

# Monitor segmentation performance
ros2 topic hz /wildscenes_segmented

# View debug output
ros2 topic echo /wildscenes_segmented --once
```

## Configuration

### ROS2 Topics

- **Input**: `/livox/lidar`
- **Output**: `/wildscenes_segmented` (segmented PointCloud2 with RGB colors)

### Segmentation Parameters

Edit `wildscenes_segmentation.py`:

```python
# Point cloud range (meters) - arrange for wider range of coverage
point_cloud_range = [-50, -3.14159265359, -10, 100, 3.14159265359, 20]

# Processing limits
max_points = 200000  # Maximum points to process per frame
min_interval = 0.5   # Minimum time between frames (seconds)
```

### Terrain Classes

The system classifies points into these semantic classes:

- **🟢 Green**: Bush, Grass, Tree-foliage
- **🟤 Brown**: Dirt, Log, Mud, Tree-trunk  
- **🟡 Yellow**: Fence
- **⚪ Gray**: Gravel, Other-terrain, Rock
- **🟠 Orange**: Structure
- **🟣 Magenta**: Object
- **⚪ White**: Unsegmented

## Performance

- **Processing Rate**: 2 FPS (0.5s per frame)
- **Points Processed**: 200K points per frame (90%+ coverage)
- **GPU Memory**: ~4-8GB CUDA memory usage
- **Latency**: ~0.3s inference + 0.2s overhead

## How It Works

1. **ROS2 Node** subscribes to `/livox/lidar` topic
2. **Preprocesses** point cloud (filtering, random sampling)
3. **Runs Cylinder3D inference** on GPU with WildScenes model
4. **Maps predictions** back to original point cloud
5. **Converts classes to RGB colors** for visualization
6. **Publishes segmented** point cloud to `/wildscenes_segmented`

## Features

- ✅ **Real-time segmentation** at 2 FPS
- ✅ **WildScenes classification** (dirt, grass, trees, fences, structures)
- ✅ **GPU acceleration** with CUDA
- ✅ **ROS2 integration** with proper message formatting
- ✅ **RViz visualization** with color-coded terrain classes
- ✅ **Docker support** for easy deployment

## Results
- Real-time terrain segmentation at 2 FPS
- GPU-accelerated inference with CUDA
- Proper ROS2 integration with PointCloud2 messages
- Color-coded visualization in RViz
- 90%+ point cloud coverage with 200K points per frame
- Stable operation with comprehensive error handling


# Returning 
This version is saved after all the process above and rviz has been tested
```bash
xhost +local:docker

docker run --rm -it \
  --network host \
  --runtime nvidia \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$HOME/.Xauthority \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=42 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:$HOME/.Xauthority:ro \
  -v $(pwd):/veggie_drive \
  veggie:foxy \
  bash
  ```

## Testing Livox SDK & ROS2 Driver 
- mostly follow steps on https://github.com/Livox-SDK/livox_ros_driver2
- will ask to set up https://github.com/Livox-SDK/Livox-SDK2/blob/master/README.md

### Livox sdk
- update samples/liv_lidar_quick_start/mid360_config.json
- then run
```
cd /veggie_drive/veggie_ws/src/Livox-SDK2
mkdir -p build && cd build
cmake ..
make

cd build/samples/livox_lidar_quick_start
./livox_lidar_quick_start ../../../samples/livox_lidar_quick_start/mid360_config.json
```

### Livox ROS Driver 2

```bash
# edit so livox publishes in format for wildscenes_node
veggie_drive/veggie_ws/src/livox_ros_driver2/launch_ROS2/msg_MID360_launch.py
xfer_format = 0

cd /veggie_drive/veggie_ws
rm -rf build install log
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash

# Install Livox SDK2 + required ROS deps
cd /veggie_drive/veggie_ws/src/Livox-SDK2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig

sudo apt-get update
sudo apt-get install -y libpcl-dev ros-foxy-pcl-conversions ros-foxy-pcl-msgs ros-foxy-rviz2

# launches Livox, WildScenes, and RealSense
ros2 launch veggie_drive_pkg wildscenes_launch.py

# Host visualization
ros2 run rviz2 rviz2 --display-config /home/unitree/Desktop/veggie_drive/segmented_points.rviz
```

## Realsense ROS Driver
- Install SDK + ROS drivers on the Jetson:
  ```bash
  sudo apt-get update
  sudo apt-get install librealsense2-utils librealsense2-dev
  sudo apt-get install ros-foxy-realsense2-camera ros-foxy-realsense2-description ros-foxy-vision-opencv
  ```
- Verify hardware:
  ```bash
  rs-enumerate-devices
  realsense-viewer 
  ```


# New running Steps
### Setup
```bash
lsusb #check for realsense

xhost +local:docker

echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
echo $RMW_IMPLEMENTATION

# if any one of them are not set or are not (42, 0, rmw_cyclonedds_cpp)
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Run Docker
```bash
  docker run --rm -it \
    --network host \
    --runtime nvidia \
    --privileged \
    --device /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0 --device /dev/video1 \
    --device /dev/video2 --device /dev/video3 \
    --device /dev/video4 --device /dev/video5 \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$HOME/.Xauthority \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ROS_DOMAIN_ID=42 \
    -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:$HOME/.Xauthority:ro \
    -v $(pwd):/veggie_drive \
    veggie:foxy bash
```
- Launch everything:
  ```bash
  cd veggie_ws 
  source install/setup.bash

  # ros bag replay
  ros2 launch veggie_drive_pkg wildscenes_launch.py 
  # live 
  ros2 launch veggie_drive_pkg wildscenes_launch.py enable_livox:=true enable_realsense:=true
  ```
- On the host, open RViz with the provided config:
  ```bash
  source /opt/ros/foxy/setup.bash

  ros2 run rviz2 rviz2 --display-config /home/unitree/Desktop/veggie_drive/segmented_points.rviz
  ```


### record bag
```
ros2 bag record \
  /livox/imu \
  /livox/lidar \
  /wildscenes_segmented \
  /camera/color/image_raw /camera/color/camera_info \
  /camera/depth/image_rect_raw /camera/depth/camera_info \
  /camera/depth/color/points


ros2 bag record -a -o rosbag_/run_$(date +%Y%m%d_%H%M%S)```

### play bag
```
ros2 bag play rosbag_/run_20251115_202031/run_20251115_202031_0.db3 --rate 4.0
```

#### calibration
```bash
## Camera (realsense) intrinisic calibration
# follow: https://docs.ros.org/en/kilted/p/camera_calibration/doc/tutorial_mono.html

sudo apt install ros-foxy-camera-calibration

# run tmux

ros2 launch realsens2_camera rs_launch.py

# measure checker board WXH (10 x 7), squares are 22.42mm (0.02)

ros2 run camera_calibration cameracalibrator --approximate 0.1 --size 8x6 --square 0.02 image:=/camera/color/image_raw camera:=camera/color

# saved to /tml/calibrationdata.tar.gz

## Lidar (Livox) extrinsic calibration
#follow : https://koide3.github.io/direct_visual_lidar_calibration/example/
```
use "veggie:calibration"

docker run --rm -it \
    --network host \
    --runtime nvidia \
    --privileged \
    --device /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0 --device /dev/video1 \
    --device /dev/video2 --device /dev/video3 \
    --device /dev/video4 --device /dev/video5 \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$HOME/.Xauthority \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ROS_DOMAIN_ID=42 \
    -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:$HOME/.Xauthority:ro \
    -v $(pwd):/veggie_drive \
    veggie:depen bash


  cd veggie_ws



export CMAKE_PREFIX PATH=$CMAKE_PREFIX_PATH:/usr/local



