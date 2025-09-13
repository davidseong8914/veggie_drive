# veggie_drive
Autonomous Navigation in the Forest


# Perception
## 1. LiDAR
- x,y,z -> WildScenes semantic segmentation
- intensity, reflectivity, penetrability, height, density
- filter => 3D Traversability

## 2. Camera
- ...


# Running this on ROS
Docker container with
- ROS2 Jazzy
- Pytorch
- CUDA


# How to run

```bash
# in wild scenes
cd /WildScenes/docker
sudo docker build -t wildscenes .

sudo docker compose run wildscenes bash # need to have Wildscenes repo cloned and added to docker.yml
nvidia-smi # check if GPU is connected
```