# Using VNC with Isaac Lab

This guide explains how to use VNC to access the Isaac Lab GUI remotely.

## Overview

Isaac Lab and its synthetic manipulation motion generation workflow require a display for the GUI components. When running on a headless server (like Brev with H100 GPUs), you need to use VNC to interact with the graphical interface.

## How to Connect

### Option 1: Web Browser Access (Recommended)

1. After running `docker compose up -d`, wait about 30 seconds for the container to fully start
2. Forward port 6080 from your Brev instance to your local machine (if not already set up in your SSH config)
   ```
   ssh -L 6080:localhost:6080 your-brev-instance
   ```
3. Open your local web browser and navigate to:
   ```
   http://localhost:6080/vnc.html
   ```
4. You should now see the Isaac Lab desktop environment

### Option 2: VNC Client

1. Forward port 5900 from your Brev instance to your local machine:
   ```
   ssh -L 5900:localhost:5900 your-brev-instance
   ```

2. Use any VNC client on your local machine to connect to:
   ```
   localhost:5900
   ```

3. No password is required (or use "password" if prompted)

## Accessing Jupyter Notebook

The Jupyter Notebook is still accessible via port 8888 as before. Forward this port and access it in your browser:

```
ssh -L 8888:localhost:8888 your-brev-instance
```

Then open:
```
http://localhost:8888/lab/tree/generate_dataset.ipynb
```

## Troubleshooting

1. **Black Screen in VNC**: Wait a few seconds as the desktop environment might take time to load.

2. **Connection Refused**: Ensure the ports are forwarded correctly and the container is running.

3. **Graphics Issues**: If you see graphics artifacts, try adjusting the VNC client settings (color depth and encoding).

4. **Container Restart**: If you need to restart the container after making changes:
   ```
   docker compose down
   docker compose up -d
   ```

5. **Isaac Lab GUI Not Loading**: Some components may take extra time to load on the first run.

## Notes

- The VNC resolution is set to 1920x1080 - you can change this in the launch.sh file if needed.
- For best performance, use the web VNC client on a fast connection.
- This setup uses noVNC (HTML5 VNC client) which works in any modern browser.