# WildScenes ROS2 Segmentation

Real-time segmentation using WildScenes Cylinder3D model in ROS2 environment. LiDAR (Livox MID-360) point cloud → WildScenes → Segmented Cloud with accurate color representation.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LiDAR Data    │    │   ROS2 Node      │    │ WildScenes      │
│                 │    │   (Python)       │    │ Cylinder3D      │
│ • Point Cloud   │───▶│ • Subscribes     │───▶│ • 3D Semantic   │
│ • Segmented     │◀───│ • Publishes      │◀───│ • GPU Accelerated│
│                 │    │ • Real-time      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

- ✅ **Real-time segmentation** at 2 FPS
- ✅ **WildScenes classification** (dirt, grass, trees, fences, structures)
- ✅ **GPU acceleration** with CUDA
- ✅ **ROS2 integration** with proper message formatting
- ✅ **RViz visualization** with color-coded terrain classes
- ✅ **Docker support** for easy deployment

## Files

- `veggie_ws/src/veggie_drive_pkg/` - ROS2 package with segmentation node
- `wildscenes/` - Model configuration and utilities
- `Dockerfile.unified` - Complete environment setup
- `segmented_points.rviz` - RViz configuration for visualization

## Quick Start

### 1. Prerequisites

- **Docker**
- **NVIDIA GPU** 
- **ROS2 Humble** 

### 2. Build and Run

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

### 3. Launch Segmentation System

```bash
# Terminal 1: Launch segmentation node
ros2 launch veggie_drive_pkg wildscenes_launch.py

# Terminal 2: Play LiDAR data (if you have bag files)
ros2 bag play data/livox_data_jetson --rate 0.5

# Terminal 3: Launch RViz for visualization
rviz2 -d segmented_points.rviz
```

### 4. Monitor Results

```bash
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





## Results
- Real-time terrain segmentation** at 2 FPS
- GPU-accelerated inference** with CUDA
- Proper ROS2 integration** with PointCloud2 messages
- Color-coded visualization** in RViz
- 90%+ point cloud coverage** with 200K points per frame
- Stable operation** with comprehensive error handling
