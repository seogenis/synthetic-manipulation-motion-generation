import os
import warp as wp
import numpy as np
from PIL import Image


# Set directories to use for inputs/outputs
ISAACLAB_OUTPUT_DIR = "_isaaclab_out"
COSMOS_OUTPUT_DIR = "_cosmos_out"

# Video and rendering settings
DEFAULT_FRAMERATE = 24.0
DEFAULT_LIGHT_DIRECTION = (0.0, 0.0, 1.0)  # Points straight down at surface


@wp.kernel
def _shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply shading to semantic segmentation using surface normals.

    Args:
        segmentation: Input semantic segmentation image (H,W,C)
        normals: Surface normal vectors (H,W,3)
        shading_out: Output shaded segmentation image (H,W,C)
        light_source: Position of light source
    """
    i, j = wp.tid()
    normal = normals[i, j]
    light_source_vec = wp.normalize(light_source[0])
    shade = 0.5 + wp.dot(wp.vec3f(normal[0], normal[1], normal[2]), light_source_vec) * 0.5

    shading_out[i, j, 0] = wp.uint8(wp.float32(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float32(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float32(segmentation[i, j, 2]) * shade)
    shading_out[i, j, 3] = wp.uint8(255)

def get_env_trial_frames(root_dir: str, camera_name: str, min_frames: int = 30) -> dict:
    """Get the last frame number for each trial for each environment in the dataset.
    
    Args:
        root_dir: Directory containing the frames
        camera_name: Name of the camera used
        min_frames: Minimum number of frames required for a valid trial
        
    Returns:
        dict: Dictionary mapping trial numbers to (start_frame, end_frame) tuples
    """
    import re
    
    # Pattern to match trial and frame numbers
    pattern = rf"{camera_name}_semantic_segmentation_trial_(\d+)_tile_(\d+)_step_(\d+).png"
    
    frames = {}
    for filename in os.listdir(root_dir):
        match = re.match(pattern, filename)
        if match:
            trial_num = int(match.group(1))
            env_num = int(match.group(2))
            frame_num = int(match.group(3))
            
            frames.setdefault(env_num, {}).setdefault(trial_num, []).append(frame_num)
            
    valid_trials = {}
    for env_num, trial_nums in sorted(frames.items()):
        for trial_num, frames in sorted(trial_nums.items()):
            # Skip if not enough frames
            if len(frames) < min_frames:
                continue
            
            # Sort frames and get range
            frames.sort()
            start_frame = frames[0]
            end_frame = frames[-1]
            
            # Verify frame sequence is continuous
            expected_frames = set(range(start_frame, end_frame + 1))
            actual_frames = set(frames)
            if len(expected_frames - actual_frames) > 0:
                continue
                
            valid_trials.setdefault(env_num, {}).setdefault(trial_num, (start_frame, end_frame))
    
    return valid_trials

def encode_video(root_dir: str, start_frame: int, num_frames: int, camera_name: str, output_path: str, env_num: int, trial_num: int) -> None:
    """CPU-based encoding of a sequence of shaded segmentation frames into a video.
    Includes both GPU and CPU fallback implementations.

    Args:
        root_dir: Directory containing the input frames
        start_frame: Starting frame index
        num_frames: Number of frames to encode
        camera_name: Name of the camera (used in filename pattern)
        output_path: Output path for the encoded video
        env_num: Environment number for the sequence
        trial_num: Trial number for the sequence

    Raises:
        ValueError: If start_frame is negative or if any required frame is missing
    """
    try:
        import cv2
    except ImportError:
        # If OpenCV is not installed, try to install it
        import subprocess
        import sys
        print("OpenCV not found. Attempting to install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        import cv2

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    frame_name_pattern = "{camera_name}_{modality}_trial_{trial_num}_tile_{env_num}_step_{frame_idx}.png"

    # Validate all frames exist before starting
    for frame_idx in range(start_frame, start_frame + num_frames):
        file_path_normals = os.path.join(root_dir, frame_name_pattern.format(camera_name=camera_name, modality="normals", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
        file_path_segmentation = os.path.join(root_dir, frame_name_pattern.format(camera_name=camera_name, modality="semantic_segmentation", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
        if not os.path.exists(file_path_normals) or not os.path.exists(file_path_segmentation):
            raise ValueError(f"Missing frame at frame index {frame_idx} for trial {trial_num}")

    # Get dimensions from first frame
    first_frame = np.array(Image.open(os.path.join(
        root_dir, frame_name_pattern.format(camera_name=camera_name, modality="semantic_segmentation", trial_num=trial_num, env_num=env_num, frame_idx=start_frame))))
    height, width = first_frame.shape[:2]
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer with MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, DEFAULT_FRAMERATE, (width, height))
    
    if not video_writer.isOpened():
        raise ValueError(f"Failed to open video writer for {output_path}")
    
    print(f"Encoding {num_frames} frames to {output_path}...")
    
    # Try to use GPU processing with warp, fall back to CPU if it fails
    use_gpu = True
    try:
        # Initialize WARP GPU resources
        normals_wp = wp.empty((height, width, 3), dtype=wp.float32, device="cuda")
        segmentation_wp = wp.empty((height, width, 4), dtype=wp.uint8, device="cuda")
        shaded_segmentation_wp = wp.empty_like(segmentation_wp)
        light_source = wp.array(DEFAULT_LIGHT_DIRECTION, dtype=wp.vec3f, device="cuda")
    except Exception as e:
        print(f"GPU processing initialization failed: {e}")
        print("Falling back to CPU-only processing...")
        use_gpu = False
    
    # Normalize light direction for CPU processing
    light_direction = np.array(DEFAULT_LIGHT_DIRECTION)
    light_direction = light_direction / np.linalg.norm(light_direction)

    for frame_idx in range(start_frame, start_frame + num_frames):
        file_path_normals = os.path.join(root_dir, frame_name_pattern.format(camera_name=camera_name, modality="normals", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
        file_path_segmentation = os.path.join(root_dir, frame_name_pattern.format(camera_name=camera_name, modality="semantic_segmentation", trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
        
        # Load frame data
        normals_np = np.array(Image.open(file_path_normals)).astype(np.float32) / 255.0
        segmentation_np = np.array(Image.open(file_path_segmentation))
        
        if use_gpu:
            try:
                # GPU-based processing with WARP
                wp.copy(normals_wp, wp.from_numpy(normals_np))
                wp.copy(segmentation_wp, wp.from_numpy(segmentation_np))
                
                # Launch kernel for shading calculation
                wp.launch(_shade_segmentation, dim=(height, width), inputs=[segmentation_wp, normals_wp, shaded_segmentation_wp, light_source])
                
                # Get the shaded image from GPU
                shaded_frame = shaded_segmentation_wp.numpy()
            except Exception as e:
                print(f"GPU processing failed on frame {frame_idx}: {e}")
                print("Switching to CPU-only processing for remaining frames...")
                use_gpu = False
                # Process this frame with CPU since GPU failed
                shaded_frame = np.zeros_like(segmentation_np)
                cpu_shade_segmentation(normals_np, segmentation_np, shaded_frame, light_direction)
        else:
            # CPU-based processing
            shaded_frame = np.zeros_like(segmentation_np)
            cpu_shade_segmentation(normals_np, segmentation_np, shaded_frame, light_direction)
        
        # OpenCV expects BGR format
        if shaded_frame.shape[2] >= 3:
            # Use only the first 3 channels (RGB/BGR)
            cv_frame = shaded_frame[:, :, :3]
            # Write frame to video
            video_writer.write(cv_frame)
        else:
            print(f"Warning: Frame has unexpected shape {shaded_frame.shape}")
    
    # Finalize video
    video_writer.release()
    print(f"Video successfully encoded to {output_path}")

def cpu_shade_segmentation(normals, segmentation, output, light_direction):
    """CPU implementation of shading calculation.
    
    Args:
        normals: Normal vectors (H,W,3) as float32 array
        segmentation: Input segmentation image (H,W,C)
        output: Output buffer for shaded result (H,W,C)
        light_direction: Normalized light direction vector
    """
    height, width = normals.shape[:2]
    channels = min(segmentation.shape[2], output.shape[2])
    
    # Parallel processing with numpy operations
    # Calculate dot product between normals and light direction
    # Reshape light_direction to (1,1,3) for broadcasting
    light = light_direction.reshape(1, 1, 3)
    dot_product = np.sum(normals * light, axis=2, keepdims=True)
    
    # Apply shading: 0.5 + dot_product * 0.5
    shading = 0.5 + 0.5 * dot_product
    
    # Apply shading to each channel
    for c in range(channels):
        output[:, :, c] = segmentation[:, :, c] * shading[:, :, 0]
    
    # Set alpha channel if it exists
    if output.shape[2] > 3 and segmentation.shape[2] > 3:
        output[:, :, 3] = segmentation[:, :, 3]
    elif output.shape[2] > 3:
        output[:, :, 3] = 255
