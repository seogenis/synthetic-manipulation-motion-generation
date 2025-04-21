# Synthetic Manipulation Motion Generation

This repository implements a two-stage data generation pipeline for robotic imitation learning:

1. **Synthetic Motion Generation** with Isaac Lab Mimic
2. **Visual Augmentation** with NVIDIA Cosmos

## System Architecture

### Docker Container

The system runs in a Docker container with these components:
- **Image**: `nvcr.io/nvidia/gr00t-smmg-bp:1.0`
- **Hardware**: Requires NVIDIA GPU(s) with CUDA support
- **Ports**:
  - 8888: JupyterLab
  - 5900: VNC Server
  - 6080: noVNC Web Access
  - 49100/tcp & 47998/udp: Livestream

The container is launched by `docker-compose.yml` and configured by `launch.sh`, which:
- Sets up a virtual display (X11)
- Configures VNC server
- Installs dependencies
- Starts JupyterLab

### Core Components

#### 1. Isaac Lab Mimic

Isaac Lab Mimic generates synthetic robot motion trajectories from a small set of human demonstrations:

- Uses annotated demonstrations from `samples/annotated_dataset.hdf5`
- Configurable parameters for randomization:
  - Robot joint state (mean, std deviation)
  - Object positions (x/y coordinates, separation)
- Outputs synthetic trajectories to `datasets/generated_dataset.hdf5`

#### 2. NVIDIA Cosmos

Cosmos applies visual transformations to the generated trajectories:

- Creates realistic renders from segmentation maps and normals
- Uses prompt-based generation with parameters:
  - Text prompt (defined in `stacking_prompt.toml`)
  - Control weight (adherence to input geometry)
  - Sigma max (amount of visual variation)
  - Canny edge detection strength

## Implementation Details

### Jupyter Notebook

The main workflow is implemented in `generate_dataset.ipynb`:

1. **Setup**: Configure environment parameters (num_envs, num_trials)
2. **Simulation**: Initialize Isaac Sim environment
3. **Parameter Adjustment**: Interactive UI for randomization controls
4. **Motion Generation**: Generate and capture synthetic trajectories
5. **Video Processing**: Create segmentation videos with shading
6. **Cosmos Integration**: Apply visual transformations

### Supporting Python Modules

- **app.py**: Flask server implementing REST API for Cosmos model
  - Endpoints: /canny/submit, /canny/status, /canny/result
  - Handles video upload, processing, and result delivery

- **cosmos_request.py**: Client for Cosmos API communication
  - Tests connection status
  - Submits processing jobs
  - Polls for completion
  - Downloads results

- **notebook_utils.py**: Utilities for data processing
  - Video encoding functions
  - Segmentation shading (using WARP)
  - Dataset frame management
  - Output path handling

- **notebook_widgets.py**: Interactive UI components
  - Parameter sliders and inputs
  - Camera selection
  - Trial number configuration

### Prompt Configuration

`stacking_prompt.toml` defines text prompts for Cosmos generation:
- Base description template
- Variable substitution for:
  - Cube materials/appearances
  - Table materials
  - Environment settings
- Negative prompts to avoid unwanted elements

## Running the System

1. Start the Docker container:
   ```
   docker-compose up
   ```

2. Access JupyterLab at http://localhost:8888

3. For visualization:
   - VNC client: localhost:5900 (password: "password")
   - Web VNC: http://localhost:6080/vnc.html

4. Run generate_dataset.ipynb end-to-end

5. For Cosmos integration:
   - Deploy Cosmos locally or remotely
   - Run app.py in Cosmos directory
   - Connect notebook to Cosmos API endpoint

## Hardware Requirements

- NVIDIA GPU with CUDA support (H100 ideal)
- Docker with NVIDIA container runtime
- Sufficient disk space for datasets and generated videos

## Known Issues

- Video encoding may fail with error `NV_ENC_ERR_UNSUPPORTED_DEVICE` on certain hardware
- Docker memory limits may need adjustment for large batch processing

## Current instance hardware
- 2 x H100 GPUs
- 128 CPU cores