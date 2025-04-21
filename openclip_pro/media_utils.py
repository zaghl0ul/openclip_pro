# -*- coding: utf-8 -*-
"""
Media utilities for video processing, clip generation, and thumbnail creation.
"""

import os
import shutil
import logging
import tempfile
import subprocess
import hashlib
import base64
import json
import io
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Configure logger
logger = logging.getLogger(__name__)

# --- Constants ---

# Standard subdirectories for project structure
PROJECT_SUBDIRS = [
    "clips",  # Generated video clips
    "thumbnails",  # Thumbnails for clips
    "exports",  # Exported final clips
    "temp",  # Temporary files (e.g., downloads)
    "frames",  # Extracted frames (if needed)
]

# FFmpeg encoding presets for generate_clips
# Maps preset names to FFmpeg command line arguments
ENCODING_PRESETS = {
    "web_optimized": [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ],
    "high_quality": [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ],
    "copy_only": ["-c:v", "copy", "-c:a", "copy"],
}
DEFAULT_ENCODING_PRESET = "web_optimized"

# --- Project Directory Management ---


def create_project_directories(base_dir: str) -> Dict[str, str]:
    """
    Create all required standard subdirectories for a project.

    Args:
        base_dir: The root directory for the project.

    Returns:
        A dictionary mapping directory keys ('base', 'clips', etc.) to their paths.

    Raises:
        OSError: If directory creation fails.
    """
    project_paths = {}
    # Create base_dir if it doesn't exist
    try:
        os.makedirs(base_dir, exist_ok=True)
        project_paths["base"] = base_dir
        logger.info(f"Ensured base project directory exists: {base_dir}")
    except OSError as e:
        logger.error(f"Failed to create base directory {base_dir}: {e}")
        raise  # Re-raise exception after logging

    # Create standard subdirectories
    for subdir in PROJECT_SUBDIRS:
        try:
            path = os.path.join(base_dir, subdir)
            os.makedirs(path, exist_ok=True)
            project_paths[subdir] = path
        except OSError as e:
            logger.error(f"Failed to create subdirectory {subdir} at {path}: {e}")
            raise  # Re-raise exception after logging

    return project_paths


def cleanup_project_directories(base_dir: str) -> bool:
    """
    Remove the entire project directory and all its contents.

    Args:
        base_dir: The root directory of the project to remove.

    Returns:
        True if cleanup was successful or directory didn't exist, False otherwise.
    """
    if not os.path.exists(base_dir):
        logger.warning(
            f"Project directory {base_dir} does not exist. Nothing to clean up."
        )
        return True  # Consider non-existence as successful cleanup

    try:
        shutil.rmtree(base_dir)
        logger.info(
            f"Successfully removed project directory {base_dir} and all its contents."
        )
        return True
    except OSError as e:  # Catch more specific OS errors if possible
        logger.error(
            f"Failed to remove project directory {base_dir}. Error: {e}"
        )
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during cleanup of {base_dir}: {e}"
        )
        return False


# --- Video Processing Functions ---


def _run_ffprobe_command(cmd: List[str]) -> Optional[Dict[str, Any]]:
    """Helper function to run ffprobe and parse JSON output."""
    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )  # Don't raise on non-zero exit

        if process.returncode != 0:
            logger.error(
                f"ffprobe command failed with exit code {process.returncode}."
                f"\nCommand: {' '.join(cmd)}\nStderr: {process.stderr.strip()}"
            )
            return None

        if not process.stdout:
            logger.warning(f"ffprobe command returned empty output. Command: {' '.join(cmd)}")
            return None

        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse ffprobe JSON output. Error: {e}"
                f"\nOutput: {process.stdout[:500]}..." # Log truncated output
            )
            return None

    except FileNotFoundError:
        logger.error(
            "ffprobe command not found. Ensure FFmpeg (including ffprobe) is installed and in the system PATH."
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while running ffprobe: {e}"
            f"\nCommand: {' '.join(cmd)}"
        )
        return None


def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata (resolution, duration, fps, etc.) from a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with metadata keys or None if an error occurred.
        Keys might include: width, height, duration, fps, codec, bitrate, size, tags_*.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    metadata = {}

    # Command to get stream info, format info, and format tags in one call
    cmd = [
        "ffprobe",
        "-v", "error",             # Only log errors
        "-show_entries",
        "stream=width,height,duration,r_frame_rate,bit_rate,codec_name", # Video stream details
        "-show_entries",
        "format=duration,size,tags",  # Format details and tags
        "-select_streams", "v:0",   # Select the first video stream
        "-of", "json",              # Output format
        video_path,
    ]

    data = _run_ffprobe_command(cmd)

    if not data:
        return None # Error logged in helper function

    # --- Extract Stream Information ---
    stream = data.get("streams", [{}])[0] # Get first stream or empty dict
    metadata["width"] = stream.get("width")
    metadata["height"] = stream.get("height")
    metadata["codec"] = stream.get("codec_name")
    metadata["bitrate"] = stream.get("bit_rate") # Often string, might need conversion

    # Parse frame rate (robustly handle "num/denom" or float string)
    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        if "/" in fps_str:
            num_str, denom_str = fps_str.split("/")
            num, denom = int(num_str), int(denom_str)
            metadata["fps"] = num / denom if denom else 0.0
        else:
            metadata["fps"] = float(fps_str)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Could not parse FPS string '{fps_str}': {e}")
        metadata["fps"] = None

    # --- Extract Format Information ---
    format_info = data.get("format", {})
    # Duration: Prefer stream duration, fallback to format duration
    stream_duration = stream.get("duration")
    format_duration = format_info.get("duration")
    try:
        if stream_duration is not None:
            metadata["duration"] = float(stream_duration)
        elif format_duration is not None:
            metadata["duration"] = float(format_duration)
        else:
            metadata["duration"] = None
    except (ValueError, TypeError):
         logger.warning(f"Could not parse duration from stream ('{stream_duration}') or format ('{format_duration}').")
         metadata["duration"] = None

    metadata["size"] = format_info.get("size") # Usually string bytes, might need int conversion
    try:
        if metadata["size"] is not None:
            metadata["size"] = int(metadata["size"])
    except (ValueError, TypeError):
        logger.warning(f"Could not convert format size '{metadata['size']}' to integer.")
        metadata["size"] = None

    # --- Extract Format Tags ---
    tags = format_info.get("tags", {})
    for key, value in tags.items():
        # Sanitize key (lowercase, replace non-alphanumeric with underscore)
        sanitized_key = "".join(c if c.isalnum() else "_" for c in key.lower()).strip("_")
        if sanitized_key: # Avoid empty keys
            metadata[f"tag_{sanitized_key}"] = value

    # Filter out None values if desired, or keep them to indicate missing data
    # metadata = {k: v for k, v in metadata.items() if v is not None}

    return metadata


def analyze_audio(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze the first audio track of a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with audio metadata or None if no audio track or error.
        Keys might include: codec, channels, sample_rate, bitrate, duration.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found for audio analysis: {video_path}")
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",  # Select the first audio stream
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bit_rate,duration",
        "-of", "json",
        video_path,
    ]

    data = _run_ffprobe_command(cmd)

    if not data:
        # Error logged in helper, might mean no audio stream or ffprobe issue
        return None

    if "streams" in data and len(data["streams"]) > 0:
        stream = data["streams"][0]
        audio_info = {
            "codec": stream.get("codec_name"),
            "channels": stream.get("channels"),
            "sample_rate": stream.get("sample_rate"), # Usually string, might need int()
            "bitrate": stream.get("bit_rate"),      # Usually string, might need int()
        }
        # Parse duration
        try:
             audio_info["duration"] = float(stream.get("duration", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse audio duration '{stream.get('duration')}'.")
            audio_info["duration"] = 0.0

        # Attempt numeric conversions where appropriate
        for key in ["channels", "sample_rate", "bitrate"]:
            try:
                if audio_info[key] is not None:
                    audio_info[key] = int(audio_info[key])
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert audio {key} '{audio_info[key]}' to integer.")
                 audio_info[key] = None # Indicate conversion failure

        return audio_info
    else:
        logger.warning(f"No audio stream found in {video_path}")
        return None # Explicitly return None if no audio stream exists


def download_youtube_video(
    url: str, quality: str = "720p", output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Download a YouTube video using yt-dlp (preferred) or youtube-dl.

    Args:
        url: The YouTube video URL.
        quality: Desired maximum video height (e.g., "720p", "1080p", "480p").
                Downloads best available quality up to this height.
        output_dir: Directory to save the downloaded file. If None, a temporary
                   directory will be created and used.

    Returns:
        Path to the downloaded video file, or None if download failed.
    """
    # Use provided dir or create a temporary one
    temp_dir_created = False
    if output_dir is None:
        try:
            output_dir = tempfile.mkdtemp(prefix="youtube_dl_")
            temp_dir_created = True
            logger.info(f"Created temporary directory for download: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create temporary directory for download: {e}")
            return None
    elif not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create specified output directory {output_dir}: {e}")
             return None

    # Generate a reasonably unique filename based on URL hash to avoid collisions
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    # Using %(title)s and %(id)s might be better if available from yt-dlp,
    # but hash is reliable if filename template fails.
    # Output template asks yt-dlp/youtube-dl to name the file.
    output_template = os.path.join(output_dir, f"video_{url_hash}_%(id)s.%(ext)s")
    # We need to find the *actual* output path later, as %(ext)s is determined by yt-dlp

    # List of command-line tools to try
    yt_tools = ["yt-dlp", "youtube-dl"]
    downloaded_path = None

    # Extract numeric height from quality string (e.g., "720p" -> "720")
    try:
        max_height = quality.lower().replace("p", "")
        int(max_height) # Validate it's a number
    except ValueError:
        logger.warning(f"Invalid quality format '{quality}'. Using default '720'.")
        max_height = "720"

    for tool in yt_tools:
        try:
            logger.info(f"Attempting download of '{url}' using {tool} (max height {max_height}p)...")
            # Command arguments for yt-dlp/youtube-dl
            cmd = [
                tool,
                # Format selection: best video up to max_height + best audio, mux into mp4 if possible
                # Falls back to best available if specific height isn't found
                "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
                "--merge-output-format", "mp4",  # Try to mux into MP4 container
                "--output", output_template,     # Output filename template
                "--no-playlist",                # Ensure only single video is downloaded if URL is part of playlist
                url,                            # The video URL
            ]

            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                # Success! Now we need to find the actual output file path.
                # yt-dlp might have named it slightly differently (e.g., .mkv initially then muxed to .mp4)
                # A simple approach is to find the first .mp4/.mkv/.webm file matching the hash pattern.
                possible_extensions = (".mp4", ".mkv", ".webm")
                found = False
                for filename in os.listdir(output_dir):
                    if url_hash in filename and filename.endswith(possible_extensions):
                        downloaded_path = os.path.join(output_dir, filename)
                        logger.info(f"Successfully downloaded and found video: {downloaded_path}")
                        found = True
                        break
                if not found:
                     # If download succeeded but we can't find the file, log error.
                     logger.error(f"{tool} reported success, but couldn't find the output file in {output_dir} matching pattern *{url_hash}*.")
                     logger.debug(f"{tool} stdout: {process.stdout}")
                     logger.debug(f"{tool} stderr: {process.stderr}")
                     downloaded_path = None # Reset path
                # If found, break the loop - we are done
                if downloaded_path:
                    break
            else:
                logger.warning(
                    f"{tool} download failed (exit code {process.returncode})."
                    f"\nStderr: {process.stderr.strip()}"
                )
                # Continue to the next tool if available

        except FileNotFoundError:
            logger.warning(f"Download tool '{tool}' not found in PATH. Trying next...")
            continue # Try the next tool
        except Exception as e:
            logger.error(f"An unexpected error occurred while trying to use {tool}: {e}")
            # Continue to the next tool if available

    if not downloaded_path:
        logger.error(f"All download attempts failed for URL: {url}")
        # Clean up the temporary directory if we created it and failed
        if temp_dir_created and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Cleaned up temporary download directory: {output_dir}")
            except OSError as e:
                logger.warning(f"Could not clean up temporary directory {output_dir}: {e}")
        return None

    return downloaded_path


def generate_clips(
    input_filepath: str,
    segments: List[Tuple[float, float]],
    output_dir: str,
    export_format: str = DEFAULT_ENCODING_PRESET,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate multiple video clips from a source video based on time segments.

    Args:
        input_filepath: Path to the source video file.
        segments: A list of tuples, where each tuple is (start_time_sec, end_time_sec).
        output_dir: Directory where the generated clips will be saved.
        export_format: The encoding preset name (key in ENCODING_PRESETS)
                       or "copy_only". Determines encoding quality/speed.
        max_workers: Maximum number of FFmpeg processes to run in parallel.

    Returns:
        A list of dictionaries, each containing info about a generated clip:
        {'index': int, 'start': float, 'end': float, 'success': bool,
         'path': Optional[str], 'duration': Optional[float], 'error': Optional[str]}
    """
    if not os.path.exists(input_filepath):
        logger.error(f"Source video for clip generation not found: {input_filepath}")
        return []

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for clips: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return []

    # Validate segments and log warnings for invalid ones
    valid_segments = []
    for i, segment in enumerate(segments):
        if not isinstance(segment, (tuple, list)) or len(segment) != 2:
            logger.warning(f"Skipping invalid segment format at index {i}: {segment}")
            continue
        start, end = segment
        try:
            start, end = float(start), float(end)
        except (ValueError, TypeError):
             logger.warning(f"Skipping segment with non-numeric times at index {i}: ({start}, {end})")
             continue

        if end <= start:
            logger.warning(
                f"Skipping invalid segment {i}: end time {end:.2f} <= start time {start:.2f}"
            )
            continue
        if start < 0:
            logger.warning(
                f"Adjusting negative start time in segment {i}: {start:.2f} -> 0.00"
            )
            start = 0.0
        valid_segments.append({"index": i, "start": start, "end": end})

    if not valid_segments:
        logger.warning("No valid segments provided for clip generation.")
        return []

    # Get encoding parameters based on the selected format
    if export_format not in ENCODING_PRESETS:
        logger.warning(
            f"Unknown export format '{export_format}'. "
            f"Falling back to default '{DEFAULT_ENCODING_PRESET}'."
        )
        export_format = DEFAULT_ENCODING_PRESET
    encoding_params = ENCODING_PRESETS[export_format]
    logger.info(f"Using encoding preset: {export_format}")


    # --- Inner function to process a single segment ---
    def process_segment(segment_info: Dict[str, Any]) -> Dict[str, Any]:
        idx = segment_info["index"]
        start_time = segment_info["start"]
        end_time = segment_info["end"]

        # Sanitize filename components
        start_str = f"{start_time:.2f}".replace(".", "_")
        end_str = f"{end_time:.2f}".replace(".", "_")
        base_name = os.path.basename(input_filepath)
        name_part = os.path.splitext(base_name)[0]
        # Basic sanitization for filename part from original video
        safe_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name_part)
        output_filename = f"{safe_name_part}_clip_{idx}_{start_str}_{end_str}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        result_data = {
            "index": idx,
            "start": start_time,
            "end": end_time,
            "success": False,
            "path": None,
            "duration": None,
            "error": None,
        }

        try:
            # Build base FFmpeg command for segment extraction
            cmd = [
                "ffmpeg",
                "-y",                   # Overwrite output files without asking
                "-ss", str(start_time), # Input seeking (fast, but potentially less accurate)
                "-i", input_filepath,
                "-to", str(end_time),   # Specify end time
                # Note: Using -ss before -i is faster for seeking but might be less frame-accurate.
                # For frame accuracy, put -ss after -i and use -copyts:
                # "-i", input_filepath, "-ss", str(start_time), "-to", str(end_time), "-copyts"
                # However, this is slower as it decodes from the beginning.
            ]

            # Add encoding parameters
            cmd.extend(encoding_params)

            # Add output file path
            cmd.append(output_path)

            # Execute FFmpeg command
            logger.debug(f"Running ffmpeg command for clip {idx}: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if process.returncode != 0:
                error_msg = f"ffmpeg error (code {process.returncode}) generating clip {idx}: {process.stderr.strip()}"
                logger.error(error_msg)
                result_data["error"] = process.stderr.strip()
                # Optionally, attempt to delete partial/failed output file
                if os.path.exists(output_path):
                    try: os.remove(output_path)
                    except OSError: pass
            else:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully generated clip {idx}: {output_path}")
                    result_data["success"] = True
                    result_data["path"] = output_path
                    result_data["duration"] = end_time - start_time
                else:
                    # Command succeeded but file is missing/empty - still an error
                    error_msg = f"ffmpeg command succeeded for clip {idx} but output file is missing or empty: {output_path}"
                    logger.error(error_msg)
                    result_data["error"] = error_msg

            return result_data

        except FileNotFoundError:
             logger.error("ffmpeg command not found. Ensure FFmpeg is installed and in the system PATH.")
             result_data["error"] = "ffmpeg not found"
             return result_data
        except Exception as e:
            logger.error(f"Unexpected error generating clip {idx}: {e}", exc_info=True)
            result_data["error"] = str(e)
            return result_data
    # --- End of inner function ---

    # Use ThreadPoolExecutor for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="clip_gen") as executor:
        # Submit all tasks
        future_to_segment = {
            executor.submit(process_segment, seg_info): seg_info
            for seg_info in valid_segments
        }

        # Collect results as they complete
        for future in future_to_segment: # Using the dictionary keys (futures)
            segment_info = future_to_segment[future]
            try:
                result = future.result() # Get result from completed future
                results.append(result)
            except Exception as e:
                # This catches errors *within the execution* of the future,
                # though process_segment should handle most internal errors.
                logger.error(
                    f"Error retrieving result for clip index {segment_info['index']}: {e}",
                    exc_info=True
                )
                results.append({
                    "index": segment_info["index"],
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                    "success": False,
                    "path": None,
                    "duration": None,
                    "error": f"Execution error: {e}",
                })

    # Sort results by original index for consistency
    results.sort(key=lambda x: x["index"])
    return results


def generate_thumbnail(
    video_path: str, time_sec: float, output_dir: str
) -> Optional[str]:
    """
    Generate a single thumbnail image from a video at a specific time.

    Args:
        video_path: Path to the source video file.
        time_sec: The time (in seconds) in the video to capture the thumbnail from.
        output_dir: Directory where the thumbnail image will be saved.

    Returns:
        The path to the generated thumbnail JPG file, or None if failed.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file for thumbnail generation not found: {video_path}")
        return None

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for thumbnails: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return None

    # Sanitize filename components
    time_str = f"{max(0, time_sec):.2f}".replace(".", "_") # Ensure time is not negative
    base_name = os.path.basename(video_path)
    name_part = os.path.splitext(base_name)[0]
    safe_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name_part)
    output_filename = f"{safe_name_part}_thumb_at_{time_str}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Build FFmpeg command for thumbnail extraction
        cmd = [
            "ffmpeg",
            "-y",                   # Overwrite if exists
            "-ss", str(time_sec),   # Seek to the specified time
            "-i", video_path,
            "-vframes", "1",        # Extract exactly one frame
            "-q:v", "2",            # Output quality for JPG (1=best, 31=worst)
            "-f", "image2",         # Force output format to image
            output_path,
        ]

        # Execute FFmpeg command
        logger.debug(f"Running ffmpeg thumbnail command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode != 0:
            logger.error(
                f"ffmpeg error (code {process.returncode}) generating thumbnail "
                f"at {time_sec:.2f}s: {process.stderr.strip()}"
            )
            # Clean up potentially empty/corrupt file
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except OSError: pass
            return None

        # Verify the output file was created and is not empty
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully generated thumbnail: {output_path}")
            return output_path
        else:
            logger.error(
                f"ffmpeg command succeeded but thumbnail file is missing or empty: {output_path}"
            )
            return None

    except FileNotFoundError:
        logger.error("ffmpeg command not found. Ensure FFmpeg is installed and in the system PATH.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating thumbnail: {e}", exc_info=True)
        return None


def optimize_and_encode_image(
    image_path: str, quality: int = 85, max_resolution: Optional[int] = 720
) -> Optional[str]:
    """
    Optimize an image (optional resize, JPEG compression) and encode it to base64.

    Requires Pillow (`pip install Pillow`) for optimization and resizing.
    If Pillow is not installed, it will only perform base64 encoding.

    Args:
        image_path: Path to the input image file.
        quality: Target JPEG quality (1-100, higher is better). Used only if Pillow is available.
        max_resolution: Maximum dimension (width or height) for resizing. If the image
                        is larger, it's resized while maintaining aspect ratio.
                        Set to None to disable resizing. Used only if Pillow is available.

    Returns:
        A base64 encoded string of the (potentially optimized) image, or None if failed.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None

    try:
        # Attempt to import Pillow
        Image = None
        try:
            from PIL import Image
        except ImportError:
            logger.warning(
                "Pillow library not found (pip install Pillow). "
                "Image will be base64 encoded without optimization or resizing."
            )

        if Image:
            # --- Pillow is available: Optimize and Resize ---
            img = Image.open(image_path)

            # Ensure image is in RGB mode for JPEG saving
            if img.mode not in ('RGB', 'L'): # L is grayscale
                 img = img.convert('RGB')

            width, height = img.size

            # Resize if needed and max_resolution is set
            if max_resolution and (width > max_resolution or height > max_resolution):
                img.thumbnail((max_resolution, max_resolution), Image.LANCZOS) # In-place resize
                new_width, new_height = img.size
                logger.debug(
                    f"Resized image from {width}x{height} to {new_width}x{new_height} "
                    f"(max_resolution: {max_resolution})"
                )

            # Save optimized image to an in-memory buffer
            buffer = io.BytesIO()
            img.save(
                buffer,
                format="JPEG",
                quality=max(1, min(quality, 100)), # Clamp quality between 1-100
                optimize=True # Enable optimization pass
                )
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            img.close() # Close the Pillow image object

        else:
            # --- Pillow not available: Read raw file bytes ---
            with open(image_path, "rb") as f:
                image_bytes = f.read()

        # Encode the image bytes (either optimized or raw) to base64
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}" # Include data URI prefix

    except FileNotFoundError: # Double check, though checked at start
         logger.error(f"Image file not found during processing: {image_path}")
         return None
    except Exception as e:
        # Catch potential Pillow errors (e.g., UnidentifiedImageError) or other issues
        logger.error(
            f"Error optimizing and encoding image {image_path}: {e}",
            exc_info=True # Include traceback in logs
        )
        return None


def generate_clip_analytics_data(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary analytics and structured data from a list of clip dictionaries.

    Note: This function assumes the input `clips` list contains dictionaries with
    specific keys like 'start', 'end', 'score', 'category', 'ai_viral_score', etc.
    Missing keys will be handled gracefully using defaults (0 or 'Unknown').

    Args:
        clips: A list of dictionaries, where each dictionary represents a clip
               and its associated metadata.

    Returns:
        A dictionary containing analytics, structured like:
        {
            "stats": { ... general statistics ... },
            "ai_stats": { ... AI-related statistics ... },
            "raw_data": [ ... list of processed clip data ... ],
            "error": Optional[str]  # Only present if an error occurred
        }
    """
    if not clips:
        logger.warning("No clip data provided for analytics generation.")
        return {"stats": {}, "ai_stats": {}, "raw_data": []}

    try:
        processed_data = []
        all_categories = []
        all_ai_tags = []
        all_ai_recommendations = []
        scores = []
        durations = []
        ai_viral_scores = []
        ai_monetization_scores = []
        total_ai_analyzed = 0

        # --- Process each clip and collect data ---
        for clip in clips:
            start = clip.get("start", 0.0)
            end = clip.get("end", 0.0)
            duration = max(0.0, end - start)
            durations.append(duration)

            score = clip.get("score", 0) # Assuming score is numeric
            scores.append(score)

            category = clip.get("category", "Unknown")
            all_categories.append(category)

            # AI related fields
            ai_viral = clip.get("ai_viral_score")
            ai_monetization = clip.get("ai_monetization_score")
            ai_tags = clip.get("ai_tags", [])
            ai_recs = clip.get("ai_recommendations", [])

            if ai_viral is not None: ai_viral_scores.append(ai_viral)
            if ai_monetization is not None: ai_monetization_scores.append(ai_monetization)
            if ai_viral is not None or ai_monetization is not None or ai_tags or ai_recs:
                total_ai_analyzed += 1

            all_ai_tags.extend(ai_tags)
            all_ai_recommendations.extend(ai_recs)

            # Store processed data for the 'raw_data' output
            processed_data.append({
                "id": clip.get("id"), # Keep original ID if present
                "start": start,
                "end": end,
                "duration": duration,
                "score": score,
                "category": category,
                "tag": clip.get("tag"), # Keep other potential fields
                "quip": clip.get("quip"),
                "clip_path": clip.get("path"), # Use 'path' from generate_clips output
                "thumbnail": clip.get("thumbnail"), # Store thumbnail path/data if available
                "ai_viral_score": ai_viral,
                "ai_monetization_score": ai_monetization,
                "ai_tags": ai_tags,
                "ai_recommendations": ai_recs,
            })

        # --- Calculate Statistics ---
        total_clips = len(clips)

        # Score Stats
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        # Duration Stats
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        total_duration = sum(durations)

        # Category Stats
        category_counts = Counter(all_categories)
        top_category = category_counts.most_common(1)[0][0] if category_counts else "N/A"

        # AI Score Stats
        avg_viral_score = (
            sum(ai_viral_scores) / len(ai_viral_scores) if ai_viral_scores else 0.0
        )
        avg_monetization_score = (
            sum(ai_monetization_scores) / len(ai_monetization_scores)
            if ai_monetization_scores
            else 0.0
        )

        # AI Tag Stats
        ai_tag_counts = Counter(all_ai_tags)
        # Get top 10 tags by frequency
        top_tags = [tag for tag, count in ai_tag_counts.most_common(10)]

        # AI Recommendation Stats
        ai_rec_counts = Counter(all_ai_recommendations)
        # Get top 5 recommendations by frequency
        top_recommendations = [rec for rec, count in ai_rec_counts.most_common(5)]


        # --- Compile Results ---
        stats = {
            "total_clips": total_clips,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "avg_duration_sec": avg_duration,
            "total_duration_sec": total_duration,
            "category_counts": dict(category_counts), # Convert Counter to dict for JSON compatibility
            "top_category": top_category,
        }

        ai_stats = {
             "total_ai_analyzed": total_ai_analyzed,
             "avg_viral_score": avg_viral_score,
             "avg_monetization_score": avg_monetization_score,
             "ai_tag_counts": dict(ai_tag_counts),
             "top_ai_tags": top_tags,
             "ai_recommendation_counts": dict(ai_rec_counts),
             "top_ai_recommendations": top_recommendations,
        }

        return {
            "stats": stats,
            "ai_stats": ai_stats,
            "raw_data": processed_data,
        }

    except Exception as e:
        logger.error(f"Error generating clip analytics data: {e}", exc_info=True)
        return {"stats": {}, "ai_stats": {}, "raw_data": [], "error": str(e)}

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Media Utils script started.")

    # Example: Create project directories
    project_dir = "./my_video_project"
    try:
        paths = create_project_directories(project_dir)
        logger.info(f"Project directories created/ensured: {paths}")

        # Example: Get metadata (replace with your video path)
        # video_file = "path/to/your/video.mp4"
        # if os.path.exists(video_file):
        #     metadata = get_video_metadata(video_file)
        #     if metadata:
        #         logger.info(f"Video Metadata:\n{json.dumps(metadata, indent=2)}")
        #     audio_meta = analyze_audio(video_file)
        #     if audio_meta:
        #         logger.info(f"Audio Metadata:\n{json.dumps(audio_meta, indent=2)}")

        # Example: Generate clips (replace with your video and segments)
        # segments_to_cut = [(10.5, 15.0), (30.0, 35.5), (60.0, 62.3)]
        # if os.path.exists(video_file):
        #     clip_results = generate_clips(video_file, segments_to_cut, paths["clips"], export_format="web_optimized")
        #     logger.info(f"Clip generation results:\n{json.dumps(clip_results, indent=2)}")

        #     # Example: Generate thumbnail for the first successful clip
        #     first_successful_clip = next((c for c in clip_results if c['success']), None)
        #     if first_successful_clip and first_successful_clip['path']:
        #          # Generate thumbnail 1 second into the clip
        #          thumb_time = 1.0
        #          thumb_path = generate_thumbnail(first_successful_clip['path'], thumb_time, paths["thumbnails"])
        #          if thumb_path:
        #              logger.info(f"Generated thumbnail: {thumb_path}")
        #              # Example: Optimize and encode thumbnail
        #              b64_thumb = optimize_and_encode_image(thumb_path, quality=80, max_resolution=360)
        #              if b64_thumb:
        #                  logger.info(f"Base64 Thumbnail (truncated): {b64_thumb[:100]}...")


    except Exception as e:
        logger.exception(f"An error occurred during example execution: {e}")
    finally:
        # Example: Clean up (optional - uncomment to remove created dirs)
        # if os.path.exists(project_dir):
        #    logger.warning(f"Cleaning up project directory: {project_dir}")
        #    if cleanup_project_directories(project_dir):
        #        logger.info("Cleanup successful.")
        #    else:
        #        logger.error("Cleanup failed.")
        pass

    logger.info("Media Utils script finished.")