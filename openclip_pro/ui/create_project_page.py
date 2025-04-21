import streamlit as st
import datetime
import os
import re
import uuid
import time
import logging
import asyncio  # Import asyncio

from database import save_project  # Use DB functions

# Use media utils for processing
from media_utils import (
    download_youtube_video,
    cleanup_project_directories,
    get_video_metadata,
    generate_clips,
    analyze_audio,
    optimize_and_encode_image,
    generate_thumbnail,
)

# Import specific tools from stdlib needed for analysis part
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
import cv2
import threading  # For lock in analysis logic

logger = logging.getLogger(__name__)


# --- Analysis Function (Refactored from original app) ---
# This function contains the core logic for analyzing chunks and calling the AI.
# It's kept here as it interacts heavily with progress bars and status messages.
# The use of asyncio.run inside a ThreadPoolExecutor worker thread is a
# non-standard pattern in Python and may have unpredictable behavior or
# performance implications in some environments, particularly within Streamlit's
# execution model. It works by creating or using a new event loop per thread
# for the async call. A more robust but complex approach would involve a
# single async loop managing all concurrent async tasks outside the thread pool.
# For alpha testing, this implementation is kept but noted as a potential area
# for future improvement if stability or performance issues arise.
def analyze_video_for_clips(
    video_path: str,
    project_id: str,  # Pass project ID for context
    temp_dirs: dict,  # Dictionary of temporary directories
    ai_module,  # Pass the initialized AI module
    analysis_config: dict,  # Pass analysis settings
    project_settings: dict,  # Pass project-level settings (quality, resolution, workers)
):
    """
    Analyze video, identify potential clips using the primary AI model.

    Args:
        video_path: Path to the video file.
        project_id: ID of the project.
        temp_dirs: Dictionary of temporary directories.
        ai_module: Initialized AIAnalysisModule instance.
        analysis_config: Dictionary containing chunk_size, frame_sample_rate, score_threshold, ai_provider, ai_model, analyze_audio.
        project_settings: Dictionary containing compression_quality, max_resolution, worker counts.

    Returns:
        Tuple: (list_of_segments_above_threshold, list_of_all_tagged_results, video_metadata_dict)
               Returns (None, None, {}) on critical failure.
    """
    start_time_analysis = time.time()
    if not ai_module or not ai_module.model_manager:
        st.error("AI Module not available. Cannot perform analysis.")
        logger.error("AI Module or Model Manager not initialized during analysis call.")
        return None, None, {}

    # --- Video Metadata ---
    video_info = get_video_metadata(video_path)
    if (
        not video_info
        or not video_info.get("duration")
        or video_info.get("duration") <= 0
    ):
        st.error(
            "Could not read essential video metadata (duration, fps). Cannot analyze."
        )
        logger.error(f"Failed to get valid metadata for video: {video_path}")
        return None, None, {}

    duration_sec = video_info["duration"]
    fps = video_info["fps"]
    if fps <= 0:
        st.error("Invalid frame rate detected. Cannot analyze.")
        logger.error(f"Invalid frame rate: {fps} for video: {video_path}")
        return None, None, video_info

    # --- Analysis Settings ---
    # Convert settings values (stored as strings) to correct types
    chunk_size = float(analysis_config.get("chunk_size", 60))
    frame_sample_rate = float(analysis_config.get("frame_sample_rate", 2.5))
    score_threshold = int(analysis_config.get("score_threshold", 75))
    primary_provider = analysis_config.get("ai_provider", "openai")
    primary_model = analysis_config.get("ai_model", "gpt-4o")
    analyze_audio_flag = bool(
        analysis_config.get("analyze_audio", False)
    )  # Assuming this setting comes from UI/defaults

    # --- Concurrency Settings ---
    # Get from project_settings (which should contain global defaults initially)
    max_workers_extraction = int(project_settings.get("max_workers_extraction", 4))
    max_workers_encoding = int(project_settings.get("max_workers_encoding", 4))
    max_workers_api = int(project_settings.get("max_workers_api", 8))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(
            f"Failed to open video file with OpenCV for chunk analysis: {video_path}"
        )
        st.error("Error: Could not open video file for analysis.")
        return None, None, video_info  # Return metadata even if analysis fails

    # --- Setup ---
    cap_lock = threading.Lock()  # Lock for thread-safe access to cv2.VideoCapture
    tagged_results_all = []

    # Determine actual chunk size and count
    actual_chunk_size = chunk_size
    if actual_chunk_size > duration_sec:
        actual_chunk_size = duration_sec
    if actual_chunk_size <= 0:
        st.error("Calculated chunk size is zero or negative. Cannot analyze.")
        cap.release()
        return None, None, video_info

    # Ensure at least one chunk
    chunk_count = max(1, int(np.ceil(duration_sec / actual_chunk_size)))

    logger.info(
        f"Analyzing video in {chunk_count} chunks of approx {actual_chunk_size:.1f}s."
    )
    if frame_sample_rate <= 0:
        logger.warning("Frame sample rate is zero or negative. Defaulting to 2.5s.")
        frame_sample_rate = 2.5

    # --- Optional Audio Analysis ---
    if analyze_audio_flag:
        analysis_status = st.status("Analyzing audio track...", expanded=False)
        with analysis_status:
            audio_info = analyze_audio(video_path)
            if audio_info:
                video_info["audio_duration"] = audio_info.get("duration")
                st.write(f"Audio duration: {audio_info.get('duration'):.1f}s")
            else:
                st.warning("Failed to analyze audio.")
            analysis_status.update(state="complete", expanded=False)

    # --- Parallel Chunk Processing ---
    progress_bar_analysis = st.progress(0.0, text="Starting analysis...")
    analysis_status_expandable = st.status(
        "Analyzing scenes...", expanded=True
    )  # Use a different variable name

    def _extract_frame_worker(frame_time_sec, chunk_idx, video_cap, lock):
        """Worker function to extract a single frame using a shared VideoCapture."""
        frame_number = int(frame_time_sec * fps)
        frame = None
        with lock:  # Thread-safe VideoCapture access
            try:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video_cap.read()
            except Exception as e:
                logger.error(
                    f"Error accessing VideoCapture for frame {frame_number} at {frame_time_sec:.2f}s in chunk {chunk_idx}: {e}"
                )
                return None  # Indicate failure

        if success and frame is not None:
            frame_filename = f"chunk_{chunk_idx}_frame_{int(frame_time_sec*1000):07d}.jpg"  # Use ms for precision
            frame_path = os.path.join(temp_dirs["frames"], frame_filename)
            try:
                # Use a reasonable quality for frames sent to AI
                cv2.imwrite(
                    frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )  # Fixed 85 quality for frames to AI
                return frame_path
            except cv2.error as e:
                logger.error(f"OpenCV error writing frame file {frame_filename}: {e}")
        else:
            logger.warning(
                f"Could not read frame at {frame_time_sec:.2f}s (frame {frame_number}) for chunk {chunk_idx}"
            )
        return None

    def _process_chunk(chunk_index, video_cap, lock):
        """Worker function to process a single video chunk."""
        chunk_start_sec = chunk_index * actual_chunk_size
        chunk_end_sec = min(chunk_start_sec + actual_chunk_size, duration_sec)
        current_chunk_duration = chunk_end_sec - chunk_start_sec
        if current_chunk_duration <= 0.1:  # Ignore very short chunks
            logger.debug(
                f"Skipping tiny chunk {chunk_index} ({current_chunk_duration:.2f}s)"
            )
            return None

        chunk_log_prefix = f"Chunk {chunk_index+1}/{chunk_count} ({chunk_start_sec:.1f}-{chunk_end_sec:.1f}s):"
        logger.info(f"{chunk_log_prefix} Starting processing.")

        # --- Frame Extraction ---
        # Sample frames uniformly across the chunk duration
        # Ensure at least one frame for short chunks, up to a reasonable limit (e.g., 10)
        num_frames_to_sample = max(
            1, min(10, int(np.ceil(current_chunk_duration / frame_sample_rate)))
        )
        # Generate time points, ensuring they are within the chunk boundaries
        frame_times_sec = np.linspace(
            chunk_start_sec, chunk_end_sec, num_frames_to_sample, endpoint=False
        ).tolist()
        # Add the very last moment if not covered (optional, but good for capturing final scene)
        # if chunk_end_sec > frame_times_sec[-1] + frame_sample_rate/2:
        #      frame_times_sec.append(chunk_end_sec - 0.01) # Sample just before end

        extracted_frame_paths = []
        # Using a *separate* ThreadPoolExecutor for frame extraction *within* chunk processing
        # might be overkill and add overhead. Consider extracting ALL frames first,
        # then processing chunks of paths. Or pass the main chunk executor.
        # Let's pass the main executor context if possible, or simplify extraction.
        # For simplicity and clarity, let's keep extraction within the chunk worker using its own executor for now.
        # Or better: just extract frames *serially* within the chunk worker as it's limited by VideoCapture lock anyway?
        # No, parallel extraction using the lock is better than serial. Let's use a small executor here.
        # Or, even better, manage frame extraction and encoding BEFORE submitting AI tasks.

        # Let's refactor: Extract and encode *all* frames first, then submit AI tasks per chunk.
        # This avoids nested ThreadPools and simplifies resource management.
        # This means `analyze_video_for_clips` needs to be restructured.

        # --- Restructuring analyze_video_for_clips ---
        # 1. Get Metadata
        # 2. Analyze Audio (Optional)
        # 3. Identify all frame sample points across the *entire* video duration.
        # 4. Extract frames in parallel using ThreadPoolExecutor and `cap_lock`.
        # 5. Encode frames to base64 in parallel using another ThreadPoolExecutor.
        # 6. Group encoded base64 images by original chunk (based on time).
        # 7. Submit AI analysis tasks for each chunk using a *third* ThreadPoolExecutor (for API calls).
        # 8. Generate Thumbnails for each chunk in parallel.
        # 9. Collect results and filter.

        # This requires significant change. For alpha, let's stick closer to the original
        # structure but simplify the _process_chunk to avoid nested executors or complex async.
        # Let's go back to the ThreadPoolExecutor per chunk, but simplify frame handling.
        # The original used a separate executor _within_ _process_chunk_ which is bad.
        # The original also used asyncio.run _within_ _process_chunk_, also potentially bad.

        # Let's keep the outer ThreadPoolExecutor for _process_chunk.
        # Inside _process_chunk, handle frame extraction and encoding serially for that chunk's frames.
        # Then perform the async AI call using asyncio.run. This keeps the structure but removes nested concurrency.

        extracted_frame_paths = []
        encoded_b64_images = []

        # Serial Frame Extraction and Encoding within this chunk's worker thread
        for frame_time_sec in frame_times_sec:
            frame_path = _extract_frame_worker(
                frame_time_sec, chunk_index, video_cap, lock
            )
            if frame_path:
                extracted_frame_paths.append(frame_path)
                # Encode immediately after extraction (still serial in this thread)
                encoded_img = optimize_and_encode_image(
                    frame_path,
                    quality=int(project_settings.get("compression_quality", 85)),
                    max_resolution=int(project_settings.get("max_resolution", 720)),
                )
                if encoded_img:
                    encoded_b64_images.append(encoded_img)
                try:
                    os.remove(frame_path)  # Clean up raw frame image immediately
                except OSError as e:
                    logger.warning(f"Could not remove raw frame file {frame_path}: {e}")

        if not encoded_b64_images:
            logger.warning(
                f"{chunk_log_prefix} No frames successfully extracted or encoded."
            )
            # Generate a placeholder result with score 0 if no images
            return {
                "id": str(uuid.uuid4()),
                "start": chunk_start_sec,
                "end": chunk_end_sec,
                "thumbnail": None,  # No thumbnail if no frames
                "score": 0,
                "tag": "No Frames",
                "quip": "Could not process frames for this segment.",
                "category": "Other",
                "colors": [],
            }

        # --- AI Analysis (Initial) ---
        # Updated prompt to include explanation for virality score
        analysis_prompt = (
            "Analyze the key visual elements, actions, and overall sentiment in this sequence of video frames "
            f"(representing time {chunk_start_sec:.1f}s to {chunk_end_sec:.1f}s). "
            "Focus on identifying potentially viral or highly engaging moments. "
            "Respond ONLY with a valid JSON object containing:\n"
            "- score: Integer (0-100) for engagement potential.\n"
            "- virality_explanation: Brief explanation (1-3 sentences) of why this clip may or may not be viral.\n"
            "- tag: Brief label (3-5 words) summarizing the scene.\n"
            "- quip: Short, witty remark (1-2 sentences) for social media.\n"
            "- category: ONE category from [Action, Dialogue, Emotional, Scenery, Humor, Informative, Tutorial, Reaction, Product, Other].\n"
            "- colors: List of 2-3 dominant color names (strings).\n\n"
            "JSON Response:"
        )

        max_retries = 2
        analysis_result_data = None
        ai_success = False
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"{chunk_log_prefix} Calling AI (Attempt {attempt+1})...")
                # --- !!! POTENTIAL ISSUE !!! ---
                # asyncio.run is called inside a ThreadPoolExecutor worker.
                # This is not the standard way to mix async and sync and can have issues.
                # A robust solution would involve managing async tasks via a single loop.
                response_text = asyncio.run(
                    ai_module.model_manager.analyze_with_model(
                        provider=primary_provider,
                        model_name=primary_model,
                        prompt=analysis_prompt,
                        images=encoded_b64_images,
                        format_type="json",  # Request JSON
                    )
                )
                logger.debug(
                    f"{chunk_log_prefix} AI response received (Attempt {attempt+1})."
                )

                # --- Response Processing & Validation ---
                if response_text is None:
                    raise ValueError("AI API returned None response.")
                if response_text.startswith("Error:"):
                    raise ValueError(f"AI API Error: {response_text}")

                # Attempt to parse JSON, handle potential prefix/suffix garbage
                try:
                    # Find the start and end of the JSON object
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}")
                    if json_start != -1 and json_end != -1:
                        clean_response = response_text[json_start : json_end + 1]
                        result_data = json.loads(clean_response)
                    else:
                        raise json.JSONDecodeError(
                            "JSON object markers not found", response_text, 0
                        )

                    # Validate required fields and types rigorously
                    score = int(result_data.get("score", 0))  # Use .get with default
                    tag = str(result_data.get("tag", "")).strip()
                    quip = str(result_data.get("quip", "")).strip()
                    virality_explanation = str(
                        result_data.get("virality_explanation", "")
                    ).strip()
                    category = str(result_data.get("category", "Other")).strip()
                    colors = [
                        str(c).strip()
                        for c in result_data.get("colors", [])
                        if isinstance(c, str)
                    ]

                    if not (0 <= score <= 100):
                        logger.warning(
                            f"{chunk_log_prefix} AI returned score out of range: {score}. Clamping to 0-100."
                        )
                        score = max(0, min(100, score))  # Clamp score

                    # Assign defaults if fields are missing/empty after stripping
                    tag = tag if tag else "Analysis Incomplete"
                    quip = quip if quip else "No quip provided."
                    virality_explanation = (
                        virality_explanation
                        if virality_explanation
                        else f"This clip has a virality score of {score}/100."
                    )

                    valid_categories = [
                        "Action",
                        "Dialogue",
                        "Emotional",
                        "Scenery",
                        "Humor",
                        "Informative",
                        "Tutorial",
                        "Reaction",
                        "Product",
                        "Other",
                    ]
                    if category not in valid_categories:
                        logger.warning(
                            f"{chunk_log_prefix} AI returned unexpected category '{category}'. Defaulting to 'Other'."
                        )
                        category = "Other"

                    analysis_result_data = {
                        "score": score,
                        "tag": tag,
                        "quip": quip,
                        "virality_explanation": virality_explanation,
                        "category": category,
                        "colors": colors,
                    }
                    ai_success = True
                    logger.info(
                        f"{chunk_log_prefix} AI analysis successful. Score: {score}"
                    )
                    break  # Success

                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.error(
                        f"{chunk_log_prefix} JSON/Validation error (Attempt {attempt+1}): {e}. Raw: '{response_text[:200]}...'"
                    )
                    if attempt == max_retries:
                        analysis_result_data = {
                            "score": 0,
                            "tag": "Analysis Error",
                            "quip": "Failed JSON Parse",
                            "virality_explanation": "Analysis could not be completed due to an error.",
                            "category": "Other",
                            "colors": [],
                        }
                    else:
                        time.sleep(1 + attempt * 1)  # Simple linear backoff
                    ai_success = False  # Mark as failure

            except (
                Exception
            ) as e:  # Catch API errors from analyze_with_model or other issues
                logger.error(
                    f"{chunk_log_prefix} Analysis failed (Attempt {attempt+1}): {e}",
                    exc_info=True,
                )
                if attempt == max_retries:
                    analysis_result_data = {
                        "score": 0,
                        "tag": "API/System Error",
                        "quip": "Analysis Failed",
                        "virality_explanation": "Analysis could not be completed due to a system error.",
                        "category": "Other",
                        "colors": [],
                    }
                else:
                    time.sleep(2 + attempt * 2)  # Simple linear backoff
                ai_success = False  # Mark as failure

        if not ai_success:
            logger.error(
                f"{chunk_log_prefix} Final AI analysis attempt failed after {max_retries+1} tries."
            )
            if (
                analysis_result_data is None
            ):  # Ensure analysis_result_data is populated on failure
                analysis_result_data = {
                    "score": 0,
                    "tag": "Analysis Failed",
                    "quip": "See logs for details",
                    "category": "Other",
                    "colors": [],
                }

        # --- Thumbnail Generation ---
        thumbnail_time_sec = chunk_start_sec + (current_chunk_duration / 2)
        thumbnail_path = generate_thumbnail(
            video_path, thumbnail_time_sec, temp_dirs["thumbnails"]
        )
        if not thumbnail_path:
            logger.warning(f"{chunk_log_prefix} Failed to generate thumbnail.")

        # --- Compile Chunk Result ---
        result_id = str(uuid.uuid4())  # Unique ID for each potential clip segment

        # Return the compiled data including analysis results
        return {
            "id": result_id,
            "start": chunk_start_sec,
            "end": chunk_end_sec,
            "duration": current_chunk_duration,  # Add duration for convenience
            "thumbnail": thumbnail_path,
            **(analysis_result_data),  # Merge analysis results
        }

    # --- Execute Chunk Processing in Parallel ---
    # Using ThreadPoolExecutor for _process_chunk
    # IMPORTANT: Pass the same VideoCapture object and lock to all workers
    # as cv2.VideoCapture is NOT thread-safe by default.
    all_chunk_results = []
    processed_count = 0

    # Need to manage the VideoCapture object lifecycle carefully with the executor
    # The cap object is created outside the executor, so it must be released AFTER it's done
    # being used by any worker thread.

    # Using the outer ThreadPoolExecutor for chunks
    with ThreadPoolExecutor(
        max_workers=max_workers_api, thread_name_prefix="ChunkAnalysis"
    ) as executor:
        # Submit tasks
        future_to_chunk = {
            executor.submit(_process_chunk, i, cap, cap_lock): i
            for i in range(chunk_count)
        }

        # Process results as they complete
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                if result:
                    all_chunk_results.append(result)
                    # logger.debug(f"Chunk {chunk_index+1} result collected.") # Too verbose?
                else:
                    logger.warning(f"Chunk {chunk_index+1} processing returned None.")
            except Exception as exc:
                logger.error(
                    f"Chunk {chunk_index + 1} generated an exception: {exc}",
                    exc_info=True,
                )
            finally:
                processed_count += 1
                # Update progress bar and status safely (Streamlit UI updates must be from main thread)
                # This update block runs in the main Streamlit thread's loop implicitly when using Streamlit's functions
                try:
                    progress = processed_count / chunk_count
                    progress_bar_analysis.progress(
                        progress,
                        text=f"Analyzing: {processed_count}/{chunk_count} chunks processed...",
                    )
                    analysis_status_expandable.update(
                        label=f"Analyzing: {processed_count}/{chunk_count} chunks processed...",
                        state="running",
                    )
                except Exception as ui_e:
                    logger.warning(
                        f"UI update failed during analysis progress: {ui_e}"
                    )  # Log potential UI update errors

    # --- Finalize Analysis ---
    # Release the VideoCapture object now that all workers are done with it
    if cap and cap.isOpened():
        cap.release()
        logger.debug("Released VideoCapture object.")

    analysis_status_expandable.update(
        label="Scene analysis complete.", state="complete", expanded=False
    )
    progress_bar_analysis.progress(1.0, text="Analysis Finished!")
    time.sleep(1)  # Let user see message
    progress_bar_analysis.empty()  # Hide the bar

    # Sort results by time position for more logical organization
    tagged_results_all = sorted(all_chunk_results, key=lambda x: x.get("start", 0))

    # Instead of filtering by threshold, we'll keep all clips regardless of score
    # We'll create segments for all analyzed chunks to generate clips for everything
    segments_for_clips = [(r["start"], r["end"]) for r in tagged_results_all]

    total_time = time.time() - start_time_analysis
    logger.info(
        f"Total analysis completed in {total_time:.2f}s. Found {len(tagged_results_all)} segments."
    )

    return segments_for_clips, tagged_results_all, video_info


# --- Streamlit Page Function ---


def show():
    """Display page for creating a new analysis project."""
    st.title("🎬 Create New Project")

    # Access the AI module instance and its registry from session state
    ai_module = st.session_state.get("ai_module")
    model_registry = st.session_state.get(
        "model_registry"
    )  # Should be set by openclip_app

    if not ai_module or not model_registry:
        st.error(
            "AI Module failed to initialize. Cannot create new projects or configure AI."
        )
        if st.button("Reload App"):
            st.rerun()
        return

    # --- Project Input and Settings ---
    col_main, col_settings = st.columns([2, 1])

    with col_main:
        # Get project name default from settings
        user_settings = st.session_state.get("user_settings", {})
        default_name_template = user_settings.get(
            "default_project_name", "Project %Y-%m-%d"
        )
        try:
            default_name = datetime.datetime.now().strftime(default_name_template)
        except Exception:  # Catch potential format errors
            default_name = "Untitled Project"
            logger.warning(
                f"Invalid default_project_name setting: {default_name_template}. Using 'Untitled Project'."
            )

        project_name = st.text_input(
            "Project Name*",
            value=default_name,
            help="Give your project a descriptive name.",
        )
        input_method = st.radio(
            "Video Source",
            ["YouTube Link", "Upload Video"],
            index=0,
            horizontal=True,
            key="input_method_radio",
        )

        uploaded_file = None
        youtube_url = None

        if input_method == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="video_uploader",
            )
            st.info(
                "Supported types: MP4, MOV, AVI, MKV, WEBM. Note: Max upload size is limited by Streamlit server configuration."
            )
        elif input_method == "YouTube Link":
            youtube_url = st.text_input(
                "Enter YouTube URL*",
                key="youtube_url_input",
                placeholder="e.g., https://www.youtube.com/watch?v=...",
                help="Paste the full YouTube video URL.",
            )

    with col_settings:
        st.subheader("Initial Analysis Parameters")
        st.caption("These are defaults; you can change global defaults in Settings.")

        # Get defaults from settings, using fallbacks
        default_clip_length = int(user_settings.get("default_clip_length", 60))
        default_frame_sample_rate = float(
            user_settings.get("default_frame_sample_rate", 2.5)
        )
        default_score_threshold = int(user_settings.get("default_score_threshold", 75))
        default_ai_provider = user_settings.get("default_ai_provider", "openai")
        default_ai_model = user_settings.get("default_ai_model", "gpt-4o")
        default_export_format = user_settings.get("export_format", "web_optimized")

        chunk_size = st.slider(
            "Approx. Segment Length (s)",
            10,
            180,
            default_clip_length,
            5,
            help="Target length for analyzed segments.",
        )
        frame_sample_rate = st.slider(
            "Frame Sample Rate (s)",
            0.5,
            10.0,
            default_frame_sample_rate,
            0.5,
            help="Frequency of frame sampling for AI vision (lower = more detail, higher AI cost).",
        )
        analyze_audio_opt = st.checkbox(
            "Include Audio Duration Analysis",
            value=False,
            help="Attempt to get audio duration using ffprobe if available.",
        )

        st.markdown("---")
        st.markdown("**Default AI Model for Analysis**")

        # Get dynamic list of providers and models from the registry
        providers = model_registry.list_providers()
        if not providers:
            st.warning("No AI providers available. Check dependencies and API keys.")
            selected_provider = "None"
            selected_model = "None"
        else:
            # Use default from settings if available, otherwise first available
            initial_provider_index = (
                providers.index(default_ai_provider)
                if default_ai_provider in providers
                else 0
            )
            selected_provider = st.selectbox(
                "Provider",
                providers,
                index=initial_provider_index,
                key="analysis_provider_select",
            )

            models = model_registry.list_models_for_provider(selected_provider)
            if not models:
                st.warning(
                    f"No models found for '{selected_provider}'. Check API keys if required."
                )
                selected_model = "None"
            else:
                # Use default model from settings if available for the selected provider
                initial_model_index = (
                    models.index(default_ai_model) if default_ai_model in models else 0
                )
                selected_model = st.selectbox(
                    "Model",
                    models,
                    index=initial_model_index,
                    key="analysis_model_select",
                )
                model_info = model_registry.get_model_info(
                    selected_provider, selected_model
                )
                if model_info and model_info.get("requires_api_key", False):
                    if not ai_module.key_manager.get_key(selected_provider):
                        st.error(
                            f"API Key required for {selected_provider.capitalize()} - {selected_model}. Please add it in '🔑 API Keys' settings."
                        )
                        # Disable analyze button if key is missing for required model

    st.divider()

    # Determine if analysis button should be enabled
    is_source_provided = (
        input_method == "Upload Video" and uploaded_file is not None
    ) or (
        input_method == "YouTube Link"
        and youtube_url is not None
        and re.match(
            r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+.*$",
            youtube_url,
        )
    )
    is_model_selected = selected_provider != "None" and selected_model != "None"

    # Check if API key is required and missing for the selected model
    api_key_missing = False
    if is_model_selected:
        model_info = model_registry.get_model_info(selected_provider, selected_model)
        if model_info and model_info.get("requires_api_key", False):
            if not ai_module.key_manager.get_key(selected_provider):
                api_key_missing = True

    analyze_button_disabled = (
        not is_source_provided or not is_model_selected or api_key_missing
    )

    if api_key_missing:
        st.warning(
            "Please set the API Key for the selected AI Provider before analyzing."
        )

    submit_button = st.button(
        "🚀 Analyze Video & Create Project",
        type="primary",
        use_container_width=True,
        key="analyze_btn",
        disabled=analyze_button_disabled,
    )

    if submit_button:
        # --- Validation ---
        if not project_name.strip():
            st.error("Project Name cannot be empty.")
            st.stop()

        source_type = ""
        source_path_for_db = ""
        input_filepath = None  # Will store the local path to the video file

        if input_method == "Upload Video":
            if uploaded_file is None:
                st.error("Please upload a video file.")
                st.stop()
            source_type = "upload"
            source_path_for_db = uploaded_file.name  # Store original name or path
        elif input_method == "YouTube Link":
            if not youtube_url:
                st.error("Please enter a YouTube URL.")
                st.stop()
            if not re.match(
                r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+.*$",
                youtube_url,
            ):
                st.error("Invalid YouTube URL format.")
                st.stop()
            source_type = "youtube"
            source_path_for_db = youtube_url  # Store the URL

        if (
            not is_model_selected or api_key_missing
        ):  # Should be caught by disabled button, but double check
            st.error("Please select a valid AI model and ensure the API key is set.")
            st.stop()

        # --- Start Processing ---
        # Generate project ID early
        project_id = str(uuid.uuid4())
        st.info(f"Starting project '{project_name}' (ID: {project_id})")
        logger.info(f"Creating new project: ID={project_id}, Name='{project_name}'")

        # 1. Create Project Directories
        # Base directory for project files will be in user's home
        base_dir = os.path.join(
            os.path.expanduser("~"), "openclip_projects", project_id
        )
        temp_dirs = {  # Define expected subdirectories
            "clips": os.path.join(base_dir, "clips"),
            "thumbnails": os.path.join(base_dir, "thumbnails"),
            "exports": os.path.join(base_dir, "exports"),
            "temp": os.path.join(
                base_dir, "temp"
            ),  # Temp files for this project (e.g., downloaded video, frames)
            "frames": os.path.join(
                base_dir, "frames"
            ),  # Directory for temporary frame images
        }
        try:
            for path in [base_dir] + list(temp_dirs.values()):
                os.makedirs(path, exist_ok=True)
            logger.info(f"Created project directories for {project_id} at {base_dir}")
        except Exception as e:
            logger.error(
                f"Failed to create project directories for {project_id}: {e}",
                exc_info=True,
            )
            st.error("Failed to create project working directories. Cannot proceed.")
            st.stop()

        cleanup_needed = True  # Flag to ensure cleanup happens on error

        try:
            # 2. Prepare Video Source
            with st.spinner("Preparing video source..."):
                if source_type == "upload":
                    # Sanitize filename to avoid issues
                    safe_filename = (
                        f"input_{re.sub(r'[^a-zA-Z0-9._-]', '_', uploaded_file.name)}"
                    )
                    input_filepath = os.path.join(temp_dirs["temp"], safe_filename)
                    try:
                        # Write the uploaded file bytes to the designated temp path within the project
                        with open(input_filepath, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        logger.info(
                            f"Uploaded file '{uploaded_file.name}' saved to {input_filepath}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to save uploaded file to project temp dir: {e}",
                            exc_info=True,
                        )
                        st.error(f"Error saving uploaded file: {e}")
                        st.stop()  # Stop here

                elif source_type == "youtube":
                    # Download YouTube video to the project's temp directory
                    input_filepath = download_youtube_video(
                        youtube_url, quality="720p", temp_dir=temp_dirs["temp"]
                    )
                    if input_filepath is None:
                        # Error already shown by download function in media_utils
                        logger.error(f"YouTube download failed for URL: {youtube_url}")
                        st.error("Failed to download YouTube video.")
                        st.stop()

            if not input_filepath or not os.path.exists(input_filepath):
                st.error(
                    "Video source file not found after preparation step. This is unexpected."
                )
                st.stop()

            logger.info(f"Video source prepared: {input_filepath}")

            # Display preview (optional and might be slow for large files)
            # try:
            #      st.subheader("Video Preview")
            #      # Streamlit handles caching for st.video
            #      st.video(input_filepath, format='video/mp4', start_time=0)
            # except Exception as e:
            #      logger.warning(f"Could not display video preview: {e}")
            #      st.warning("Could not display video preview.")

            # 3. Perform Initial AI Analysis (using the refactored function)
            st.info("Starting initial AI analysis... This may take several minutes.")

            # Collect analysis and project settings to pass to the analysis function
            analysis_params = {
                "chunk_size": chunk_size,
                "frame_sample_rate": frame_sample_rate,
                "score_threshold": score_threshold,
                "ai_provider": selected_provider,  # Pass selected provider
                "ai_model": selected_model,  # Pass selected model
                "analyze_audio": analyze_audio_opt,
            }
            # Get all worker settings from user_settings for project_settings
            project_params = {
                "compression_quality": user_settings.get("compression_quality", "85"),
                "max_resolution": user_settings.get("max_resolution", "720"),
                "max_workers_extraction": user_settings.get(
                    "max_workers_extraction", "4"
                ),
                "max_workers_encoding": user_settings.get("max_workers_encoding", "4"),
                "max_workers_api": user_settings.get("max_workers_api", "8"),
                "max_workers_clip_gen": user_settings.get("max_workers_clip_gen", "4"),
                "export_format": default_export_format,  # Pass default export format
            }

            segments_above_threshold, tagged_results_all, video_info = (
                analyze_video_for_clips(
                    video_path=input_filepath,
                    project_id=project_id,
                    temp_dirs=temp_dirs,
                    ai_module=ai_module,
                    analysis_config=analysis_params,
                    project_settings=project_params,  # Pass worker settings here
                )
            )

            if (
                tagged_results_all is None
            ):  # analyze_video_for_clips returns None, None, {} on critical failure
                st.error("Video analysis failed critically. Cannot proceed.")
                # No need to stop, just don't generate clips or save project with analysis results
                tagged_results_all = []  # Ensure it's an empty list if analysis failed
                segments_above_threshold = []  # Ensure it's empty

            if not tagged_results_all:
                st.warning(
                    "Initial AI analysis did not identify any significant segments."
                )
                # Continue to save the project with empty clips list but maybe some video info

            # 4. Generate Clip Files for all segments
            generated_clips_info = []
            if segments_for_clips:  # formerly segments_above_threshold
                st.info(f"Generating {len(segments_for_clips)} clip files...")
                # Pass input_filepath and output_dir from the project's directories
                # Use export_format from settings and workers from project_params
                generated_clips_info = generate_clips(
                    input_filepath=input_filepath,
                    segments=segments_for_clips,  # now includes all clips regardless of score
                    output_dir=temp_dirs["clips"],
                    export_format=project_params.get("export_format", "web_optimized"),
                    max_workers=int(project_params.get("max_workers_clip_gen", 4)),
                )
                # Note: generate_clips function handles its own messages and logs
                if not generated_clips_info:
                    st.warning(
                        "Clip file generation finished, but no files were successfully created."
                    )

            else:
                st.info("No segments were found. Cannot generate any clips.")

            # 5. Prepare Full Project Data for Saving
            # Start with the results from analysis, which are the basis for clips
            clips_for_db = tagged_results_all
            # Map generated clip file paths to the correct clip entries based on start/end time
            generated_clips_map = {
                (round(c["start"], 2), round(c["end"], 2)): c["path"]
                for c in generated_clips_info
                if c.get("start") is not None and c.get("end") is not None
            }

            # Update clips_for_db with the clip_path if the segment had a file generated
            for clip_entry in clips_for_db:
                clip_key = (
                    round(clip_entry.get("start", -1), 2),
                    round(clip_entry.get("end", -1), 2),
                )
                # Check if this clip segment had a corresponding file generated
                if clip_key in generated_clips_map:
                    clip_entry["clip_path"] = generated_clips_map[clip_key]
                else:
                    # Ensure clip_path is None if no file was generated (e.g., score below threshold)
                    clip_entry["clip_path"] = None

            # Get the highest scoring clip ID to store in the project table for quick lookup
            top_clip_id = None
            if clips_for_db:
                # Find the clip with the highest score among all analyzed segments
                top_clip = max(
                    clips_for_db, key=lambda c: c.get("score", 0), default=None
                )
                if top_clip and top_clip.get("id"):
                    top_clip_id = top_clip["id"]
                elif top_clip:
                    logger.warning(
                        f"Top clip found but is missing ID: {top_clip}. Cannot set top_clip_id."
                    )

            creation_time_iso = datetime.datetime.now().isoformat()

            project_data = {
                "id": project_id,
                "name": project_name.strip(),
                "created_at": creation_time_iso,
                "source_type": source_type,
                "source_path": source_path_for_db,  # Original user input path/URL
                "base_dir_path": base_dir,  # Path to the project's directory on disk
                "settings": {
                    **analysis_params,
                    **project_params,
                },  # Store combined settings used for this project
                "video_info": video_info,
                "top_clip_id": top_clip_id,  # Add the top clip ID to project data
                "clips": clips_for_db,  # Add the list of clip dictionaries
            }

            # 6. Save Project to Database
            # The save_project function handles saving project record and clips,
            # and it expects the list of clips in the input dict.
            if save_project(project_data):
                st.success(f"Project '{project_name}' created successfully!")
                st.balloons()
                cleanup_needed = False  # Don't cleanup temp project dir if successful

                # Navigate to the new project details page
                st.session_state.current_project_id = project_id
                st.session_state.app_mode = "📁 Projects"
                st.rerun()
            else:
                st.error("Failed to save project details to the database.")
                # Keep cleanup_needed = True so the temp project dir is removed

        except Exception as e:
            logger.error(
                f"Error during project creation pipeline for {project_id}: {e}",
                exc_info=True,
            )
            st.error(f"An unexpected error occurred: {e}")
        finally:
            # 7. Cleanup Temp Dirs if Error Occurred or Process Interrupted
            # Note: cleanup_project_directories deletes the base_dir.
            # If cleanup_needed is False, the project was saved, and its files should persist.
            # If cleanup_needed is True, something failed, and we should remove the partially created directory.
            if cleanup_needed and base_dir and os.path.exists(base_dir):
                logger.warning(
                    f"Attempting cleanup for project {project_id} due to error or incomplete process."
                )
                try:
                    # Attempt to clean up the entire base directory if project saving failed
                    cleanup_project_directories(base_dir)
                    st.info(
                        f"Cleaned up incomplete project files for '{project_name}'."
                    )
                except Exception as cleanup_e:
                    logger.error(
                        f"Failed final cleanup for {project_id}: {cleanup_e}",
                        exc_info=True,
                    )
                    st.error(
                        f"Failed to fully clean up temporary project files: {cleanup_e}"
                    )

    # Remove conflicting/unused functions
    # del fetch_models_from_provider_cached # These were never used correctly
    # del fetch_models_from_provider
    # del query_provider_a_api
    # del query_provider_b_api
