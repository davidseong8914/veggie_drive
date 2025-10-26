# WildScenes ROS2 Segmentation
Real-time segmentation using WildScenes Cylinder3D model in ROS2 environment. LiDAR (Livox MID-360) point cloud → WildScenes → Segmented Cloud with accurate color representation.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LiDAR Data    │    │   ROS2 Node      │    │ WildScenes      │
│                 │    │         │    │ Cylinder3D      │
│ • Point Cloud   │───▶│ • Subscribes     │───▶│ • 3D Semantic segmentation  │
│ • Segmented pointcloud   │◀───│ • Publishes      │◀───│ 
│                 │    │       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files
- `veggie_ws/src/veggie_drive_pkg/` - ROS2 package with segmentation node
- `wildscenes/` - Model configuration and utilities
- `Dockerfile.unified` - Docker environment for x86
- 'Docker 
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
docker build -f Dockerfile.jetson.foxy -t foxy:jetson2 .

docker run --rm -it --network host --runtime=nvidia \
  -e DISPLAY -e XAUTHORITY -e QT_X11_NO_MITSHM=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e USER=$USER -e HOME=$HOME \
  -e XDG_RUNTIME_DIR=/tmp/runtime-$(id -u) \
  -e ROS_LOG_DIR=$HOME/.ros/log \
  -e ROS_DISTRO=foxy \
  -v ${XAUTHORITY:-$HOME/.Xauthority}:${XAUTHORITY:-$HOME/.Xauthority}:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/dri:/dev/dri \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v $(pwd):/veggie_drive \
  foxy:jetson bash

cd /veggie_drive/veggie_ws
source /opt/ros/foxy/install/setup.bash
rm -rf build install log
colcon build --packages-select veggie_drive_pkg
source install/setup.bash
```

### 2.3 Circular import fix
```
pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true

apt-get update && apt-get install -y --no-install-recommends \
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
ros2 bag play data/livox_data_jetson_0.db3 --rate 0.5
ros2 bag play data/livox_data_jetson_0.db3 --rate 1.0


# Terminal 3: Launch RViz for visualization
rviz2 -d segmented_points.rviz
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
# Point cloud range (meters)
point_cloud_range = [-50, -3.14159265359, -10, 50, 3.14159265359, 20]

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
- Real-time terrain segmentation** at 2 FPS
- GPU-accelerated inference** with CUDA
- Proper ROS2 integration** with PointCloud2 messages
- Color-coded visualization** in RViz
- 90%+ point cloud coverage** with 200K points per frame
- Stable operation** with comprehensive error handling



veg_fox:1025 - Dockefile.jetson.optimized... 
  - this failed and is definitely not the way to go
  Dockerfile.jetson.optimized (X) wrong
  and 
  Dockerfile.jetson.wildscenes (X) not yet

foxy:jetson - copy of Dockerfile.jetson with foxy 
  

## rviz error 
// on the jetson
export DISPLAY=localhost:10.0
set ROS_DOMAI_ID=42 // arbitrary value but different than system // this was also added to ~/.bashrc


xhost +local:docker // run on the host machine

export DISPLAY=localhost:10.0
export XAUTH=/tmp/.docker.xauth

docker run --rm -it --network host --runtime=nvidia \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTH \
  -e QT_X11_NO_MITSHM=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e USER=$USER -e HOME=$HOME \
  -e XDG_RUNTIME_DIR=/tmp/runtime-$(id -u) \
  -e ROS_LOG_DIR=$HOME/.ros/log \
  -e ROS_DISTRO=foxy \
  -e ROS_DOMAIN_ID=42 \
  -v $XAUTH:$XAUTH:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/dri:/dev/dri \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v $HOME:$HOME \
  -v $(pwd):/veggie_drive \
  foxy:jetson bash

export DISPLAY=localhost:10.0
apt update && apt install -y ros-foxy-rviz2
rviz2

---

## Setting Up X11 Forwarding for RViz2 on a New Laptop

If SSH'ing into the Jetson from a new laptop to use RViz2:

### On Your Local Laptop (one-time setup):
```bash
# Allow Docker containers to access X11
xhost +local:docker

# Reconnect to Jetson with X11 forwarding
ssh -X unitree@<jetson-ip>
```

### On the Jetson (in SSH session):
```bash
# Set display and ROS domain (already in ~/.bashrc on this Jetson)
export DISPLAY=localhost:10.0
export ROS_DOMAIN_ID=42

# Start Docker with proper X11 configuration
export XAUTH=$HOME/.docker.xauth
rm -f $XAUTH && touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run --rm -it --network host --runtime=nvidia \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=42 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v $HOME:$HOME \
  -v $(pwd):/veggie_drive \
  foxy:jetson bash

# Inside Docker container:
apt update && apt install -y ros-foxy-rviz2  # First time only
source /opt/ros/foxy/setup.bash
rviz2
```

**Notes:**
- `ROS_DOMAIN_ID=42` isolates ROS2 communication from other processes
- Must use `ssh -X` (capital X) not `ssh -x` (lowercase)
- `xhost +local:docker` must run on your local machine, not the Jetson