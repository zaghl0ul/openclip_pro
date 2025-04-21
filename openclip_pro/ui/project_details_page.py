import streamlit as st
import os
import time
import datetime
import logging
import pandas as pd
from matplotlib import pyplot as plt
import io
import base64
import numpy as np
import sqlite3
from typing import List, Dict, Optional, Any

from database import load_project, delete_project, DB_FILE
from media_utils import (
    cleanup_project_directories,
    generate_clips,
    generate_clip_analytics_data as generate_base_analytics,
)
from ui.components import clip_display

# Import AI UI components and the AIAnalysisModule structure if needed for AI tab
# from ai.ui_components import display_ai_board_ui # Example
# from ai.ai_models import AIAnalysisModule # Example

logger = logging.getLogger(__name__)

# --- Constants ---
CLIP_SORT_OPTIONS = [
    "Score (High-Low)",
    "Score (Low-High)",
    "Duration",
    "Position",
    "AI Viral Score",
    "AI Monet. Score",
]
CLIP_VIEW_MODES = ["Gallery", "List", "Timeline"]
EXPORT_PLATFORMS = ["Standard Web", "Instagram", "TikTok", "Twitter", "Original"]
EXPORT_FORMATS = ["web_optimized", "high_quality", "copy_only"]

# --- Helper function for Analytics Plot ---


def _create_analytics_chart(
    clips: List[Dict[str, Any]], video_duration: Optional[float] = None
) -> Optional[str]:
    """
    Generates a Matplotlib visualization summarizing clip analytics.

    Args:
        clips: A list of clip dictionaries.
        video_duration: The total duration of the source video in seconds.

    Returns:
        A base64 encoded string of the generated PNG image, or None if an error occurs
        or no clips are provided.
    """
    if not clips:
        logger.info("No clips provided for analytics chart generation.")
        return None

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10), facecolor="#2E2E2E"
        )  # Dark background

        # --- Data Preparation ---
        scores = [c.get("score", 0) for c in clips]
        # Calculate durations safely
        # durations = [max(0, c.get("end", 0) - c.get("start", 0)) for c in clips] # Unused

        # Tally categories
        categories = {}
        for clip in clips:
            cat = clip.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

        # Calculate average score per category
        cat_scores: Dict[str, List[float]] = {}
        for clip in clips:
            cat = clip.get("category", "Unknown")
            cat_scores.setdefault(cat, []).append(clip.get("score", 0))

        avg_score_by_category = {
            cat: sum(score_list) / len(score_list) if score_list else 0
            for cat, score_list in cat_scores.items()
        }

        # Prepare timeline data (using clip start time)
        timeline_data_points = sorted(
            [
                {"time": c.get("start", 0), "score": c.get("score", 0)}
                for c in clips
                if c.get("start") is not None
            ],
            key=lambda x: x["time"],
        )

        # --- Plotting Configuration ---
        plot_face_color = "#3B3B3B"
        text_color = "white"
        grid_color = "gray"
        grid_alpha = 0.4
        axis_color = "white"

        # Function to style axes consistently
        def style_ax(ax):
            ax.set_facecolor(plot_face_color)
            ax.tick_params(colors=text_color, which="both")
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_color(axis_color)

        # --- Plot 1: Score Distribution ---
        ax = axes[0, 0]
        if scores:
            ax.hist(
                scores,
                bins=range(0, 101, 10),
                color="#4CAF50",
                alpha=0.8,
                edgecolor="white",
            )
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
        else:
            ax.text(
                0.5,
                0.5,
                "No Score Data",
                ha="center",
                va="center",
                color=text_color,
                transform=ax.transAxes,
            )
        ax.set_title("Initial Score Distribution")
        style_ax(ax)

        # --- Plot 2: Category Pie Chart ---
        ax = axes[0, 1]
        if categories:
            sorted_categories = sorted(categories.items(), key=lambda item: item[1], reverse=True)
            labels = [item[0] for item in sorted_categories]
            sizes = [item[1] for item in sorted_categories]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=None,  # Labels are added to legend
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.85,
                colors=plt.cm.viridis(np.linspace(0, 1, len(labels))),
            )
            ax.legend(
                wedges,
                labels,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5), # Adjust legend position
                labelcolor=text_color,
                facecolor=plot_face_color,
                edgecolor=axis_color
            )
            plt.setp(autotexts, size=10, weight="bold", color="white")
        else:
            ax.text(
                0.5,
                0.5,
                "No Category Data",
                ha="center",
                va="center",
                color=text_color,
                transform=ax.transAxes,
            )
        ax.set_title("Content Categories")
        style_ax(ax)
        # Remove axis labels/ticks for pie chart
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # --- Plot 3: Clip Timeline (Score over time) ---
        ax = axes[1, 0]
        if timeline_data_points:
            times = [item["time"] for item in timeline_data_points]
            scores_tl = [item["score"] for item in timeline_data_points]
            ax.plot(
                times,
                scores_tl,
                color="#FF9800",
                marker="o",
                linestyle="-",
                linewidth=2,
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score")
            ax.grid(True, linestyle=":", alpha=grid_alpha, color=grid_color)
            if video_duration is not None and video_duration > 0:
                ax.set_xlim(0, video_duration)
            ax.set_ylim(0, 100)
        else:
            ax.text(
                0.5,
                0.5,
                "No Timeline Data",
                ha="center",
                va="center",
                color=text_color,
                transform=ax.transAxes,
            )
        ax.set_title("Initial Score Timeline")
        style_ax(ax)

        # --- Plot 4: Average Score by Category ---
        ax = axes[1, 1]
        if avg_score_by_category:
            sorted_cats = sorted(
                avg_score_by_category.items(), key=lambda item: item[1], reverse=True
            )
            cats_labels = [item[0] for item in sorted_cats]
            avg_scores_values = [item[1] for item in sorted_cats]
            bars = ax.bar(
                cats_labels,
                avg_scores_values,
                color=plt.cm.plasma(np.linspace(0, 1, len(cats_labels))),
                alpha=0.8,
            )
            ax.tick_params(axis="x", rotation=45)
            ax.yaxis.grid(True, linestyle=":", alpha=grid_alpha, color=grid_color)
            ax.set_ylabel("Avg Score")
            ax.set_title("Average Initial Score by Category")
            ax.set_ylim(0, 100)
            plt.setp(ax.get_xticklabels(), ha="right") # Improve rotated label alignment
        else:
            ax.text(
                0.5,
                0.5,
                "Not Enough Data",
                ha="center",
                va="center",
                color=text_color,
                transform=ax.transAxes,
            )
            ax.set_title("Average Initial Score by Category")

        style_ax(ax)

        # --- Final Touches & Export ---
        plt.tight_layout(pad=3.0, rect=[0, 0, 0.9, 1]) # Adjust layout slightly for legend
        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", dpi=150, facecolor=fig.get_facecolor()
        )
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)  # Close the figure to free memory
        return img_str
    except Exception as e:
        logger.error(f"Error generating analytics plot: {e}", exc_info=True)
        st.warning(f"Could not generate analytics visualization: {e}")
        return None


# --- Tab Content Functions ---


def _show_clips_tab(project: Dict[str, Any]) -> None:
    """
    Displays the clips associated with a project, including filtering,
    sorting, and different view modes.
    """
    st.subheader("🎥 Project Clips")
    clips = project.get("clips", [])
    project_id = project["id"]

    if not clips:
        st.info(
            "No clips were identified during the initial analysis for this project."
        )
        return

    # Use session state to store filter/sort/view settings for this specific project's clips tab
    # This preserves the view state during interactions within the tab.
    state_key = f"clips_view_state_{project_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "min_score": 0,
            "category_filter": [],
            "sort_by": CLIP_SORT_OPTIONS[0],  # Default: Score (High-Low)
            "view_mode": CLIP_VIEW_MODES[0],  # Default: Gallery
        }
        logger.debug(f"Initialized clip view state for project {project_id}")

    # Retrieve current state values
    current_state = st.session_state[state_key]

    # --- Filtering & Sorting Controls ---
    filter_cols = st.columns([1, 2, 1, 2])
    with filter_cols[0]:
        # Update state directly from widget interaction
        current_state["min_score"] = st.slider(
            "Min Initial Score",
            0,
            100,
            current_state["min_score"],
            key=f"clip_min_score_{project_id}",
        )

    with filter_cols[1]:
        available_categories = sorted(
            list(set(c.get("category", "Unknown") for c in clips if c.get("category")))
        )
        current_state["category_filter"] = st.multiselect(
            "Filter Category",
            options=available_categories,
            default=current_state["category_filter"],
            key=f"clip_cat_filter_{project_id}",
        )

    with filter_cols[2]:
        try:
            sort_index = CLIP_SORT_OPTIONS.index(current_state["sort_by"])
        except ValueError:
            sort_index = 0 # Default to first option if saved state is invalid
        current_state["sort_by"] = st.selectbox(
            "Sort by",
            CLIP_SORT_OPTIONS,
            index=sort_index,
            key=f"clip_sort_{project_id}",
        )

    with filter_cols[3]:
        try:
            view_index = CLIP_VIEW_MODES.index(current_state["view_mode"])
        except ValueError:
            view_index = 0 # Default to first option if saved state is invalid
        current_state["view_mode"] = st.radio(
            "View Mode",
            CLIP_VIEW_MODES,
            index=view_index,
            horizontal=True,
            key=f"clip_view_mode_{project_id}",
        )

    # --- Apply Filtering ---
    filtered_clips = [
        c for c in clips if c.get("score", 0) >= current_state["min_score"]
    ]
    if current_state["category_filter"]:
        filtered_clips = [
            c
            for c in filtered_clips
            if c.get("category", "Unknown") in current_state["category_filter"]
        ]

    # --- Apply Sorting ---
    sort_by = current_state["sort_by"]
    reverse_sort = True # Default high-to-low

    # Define sorting key function safely, handling potential None/missing values
    def get_sort_key(clip: Dict[str, Any]) -> Any:
        if sort_by == "Score (Low-High)":
            return clip.get("score", 0)
        elif sort_by == "Duration":
            return max(0, clip.get("end", 0) - clip.get("start", 0))
        elif sort_by == "Position":
            return clip.get("start", 0)
        elif sort_by == "AI Viral Score":
            # Sort missing (-1 or None) values last when sorting high-to-low
            return clip.get("ai_viral_score", -1)
        elif sort_by == "AI Monet. Score":
            return clip.get("ai_monetization_score", -1)
        # Default is "Score (High-Low)"
        return clip.get("score", 0)

    if sort_by in ["Score (Low-High)", "Position"]:
        reverse_sort = False

    try:
        # Add a secondary sort key (e.g., start time) for stable sorting
        filtered_clips.sort(key=lambda c: (get_sort_key(c), c.get("start", 0)), reverse=reverse_sort)
    except Exception as e:
        logger.error(f"Sorting failed for project {project_id}: {e}", exc_info=True)
        st.error(f"Sorting failed: {e}")
        # Keep the merely filtered list if sorting fails

    st.caption(f"Showing {len(filtered_clips)} of {len(clips)} clips.")
    st.markdown("---")

    if not filtered_clips:
        st.info("No clips match the current filter settings.")
        return

    # --- Display based on View Mode ---
    view_mode = current_state["view_mode"]
    if view_mode == "Gallery":
        clip_display.display_clips_gallery(filtered_clips)
    elif view_mode == "List":
        clip_display.display_clips_list(filtered_clips)
    elif view_mode == "Timeline":
        video_duration = project.get("video_info", {}).get("duration")
        clip_display.display_clips_timeline(filtered_clips, video_duration, sort_by)


def _show_analytics_tab(project: Dict[str, Any], ai_module: Optional[Any]) -> None:
    """
    Displays project analytics, including summary statistics, visualizations,
    and raw clip data. Optionally integrates AI-enhanced analytics.
    """
    st.subheader("📊 Project Analytics")
    clips = project.get("clips", [])

    if not clips:
        st.info("No clips available to generate analytics for this project.")
        return

    # --- Generate Analytics Data ---
    analytics_data = generate_base_analytics(clips)

    # Attempt to generate enhanced analytics if AI module is available and capable
    if ai_module and hasattr(ai_module, "generate_enhanced_analytics"):
        try:
            enhanced_data = ai_module.generate_enhanced_analytics(
                clips, project, analytics_data.copy() # Pass a copy in case it modifies
            )
            if enhanced_data: # Use enhanced data only if successfully generated
                analytics_data = enhanced_data
                logger.debug(f"Using enhanced analytics for project {project['id']}")
            else:
                logger.warning(f"AI enhanced analytics returned no data for project {project['id']}")
        except Exception as ai_analytics_err:
            logger.error(
                f"Error generating AI enhanced analytics for project {project['id']}: {ai_analytics_err}",
                exc_info=True,
            )
            st.warning(f"Could not generate AI enhanced analytics: {ai_analytics_err}")
            # Fallback: analytics_data already holds the base analytics

    if not analytics_data:
        st.warning("Analytics generation failed or returned no data.")
        return

    # --- Display Stats ---
    st.markdown("### Summary Statistics")
    stats = analytics_data.get("stats", {})
    ai_stats = analytics_data.get("ai_stats", {}) # Get AI stats if generated

    stat_cols = st.columns(4)
    stat_cols[0].metric("Clips Identified", stats.get("total_clips", 0))
    stat_cols[1].metric("Avg Initial Score", f"{stats.get('avg_score', 0):.1f}")
    stat_cols[2].metric("Avg Duration", f"{stats.get('avg_duration', 0):.1f}s")
    stat_cols[3].metric("Top Category", stats.get("top_category", "N/A"))

    # Display AI stats only if they exist and contain meaningful data
    if ai_stats:
        st.markdown("#### AI Analysis Insights")
        ai_stat_cols = st.columns(4)

        # Check for existence and non-zero/non-default values before displaying metrics
        avg_viral = ai_stats.get('avg_viral_score')
        if avg_viral is not None and avg_viral > -1: # Assuming -1 is placeholder
             ai_stat_cols[0].metric("Avg Viral Score", f"{avg_viral:.1f}")
        else:
             ai_stat_cols[0].metric("Avg Viral Score", "N/A")

        avg_monetize = ai_stats.get('avg_monetization_score')
        if avg_monetize is not None and avg_monetize > -1:
             ai_stat_cols[1].metric("Avg Monetize Score", f"{avg_monetize:.1f}")
        else:
             ai_stat_cols[1].metric("Avg Monetize Score", "N/A")


        with ai_stat_cols[2]:
            st.markdown("**Top AI Tags:**")
            tags = ai_stats.get("top_tags", [])
            st.caption(
                "<br>".join(f"- {tag}" for tag in tags[:5]) if tags else "N/A",
                unsafe_allow_html=True,
            )
        with ai_stat_cols[3]:
            st.markdown("**Top AI Recs:**")
            recs = ai_stats.get("top_recommendations", [])
            # Truncate long recommendations
            st.caption(
                "<br>".join(f"- {rec[:50]}{'...' if len(rec) > 50 else ''}" for rec in recs[:3]) if recs else "N/A",
                unsafe_allow_html=True,
            )

    # --- Display Charts ---
    st.markdown("### Visualizations")
    video_duration = project.get("video_info", {}).get("duration")
    base_chart_b64 = _create_analytics_chart(clips, video_duration)
    if base_chart_b64:
        st.image(
            f"data:image/png;base64,{base_chart_b64}",
            use_column_width=True,
            caption="Clip Analytics Overview",
        )
    else:
        st.info("Base analytics chart could not be generated.")

    # Display AI-enhanced chart if available
    ai_chart_b64 = analytics_data.get("ai_chart") # Assuming AI module adds this key
    if ai_chart_b64:
        st.image(
            f"data:image/png;base64,{ai_chart_b64}",
            use_column_width=True,
            caption="AI-Enhanced Analytics View",
        )

    # --- Raw Data ---
    st.markdown("### Raw Data")
    with st.expander("View Raw Clip Data"):
        # Use the raw clip data (original list of dicts) for display
        if clips:
            try:
                df_raw = pd.DataFrame(clips)
                # Define preferred columns order, excluding potentially large/complex ones by default
                default_display_cols = [
                    "id", "start", "end", "score", "category", "tag", "quip",
                    "ai_viral_score", "ai_monetization_score", "ai_tags", "ai_recommendations"
                ]
                # Filter to only show columns that actually exist in the DataFrame
                cols_to_show = [col for col in default_display_cols if col in df_raw.columns]
                # Add any remaining columns not in the default list to the end
                other_cols = [col for col in df_raw.columns if col not in cols_to_show and col != 'data'] # Exclude 'data'
                cols_to_show.extend(other_cols)

                st.dataframe(df_raw[cols_to_show], use_container_width=True)
            except ImportError:
                 st.warning("Pandas library not available. Displaying raw JSON.")
                 st.json(clips[:10]) # Show sample as JSON
            except Exception as e:
                logger.error(f"Failed to display raw clip data for project {project['id']}: {e}", exc_info=True)
                st.error(f"Failed to display raw data: {e}")
                st.json(clips[:10]) # Fallback to JSON
        else:
            st.info("No raw clip data available.")


def _find_source_video(project_base_dir: Optional[str], project_id: str) -> Optional[str]:
    """Locates the likely source video file within the project's temp directory."""
    if not project_base_dir:
        logger.warning(f"Project base directory not provided for project {project_id}.")
        return None

    temp_dir_path = os.path.join(project_base_dir, "temp")
    if not os.path.isdir(temp_dir_path):
        logger.warning(f"Project temp directory not found for project {project_id}: {temp_dir_path}")
        return None

    # Look for files potentially created during project setup
    # This assumption might be brittle if naming conventions change.
    possible_inputs = [
        f for f in os.listdir(temp_dir_path)
        if f.startswith("input_") or f.startswith("downloaded_video")
    ]

    if not possible_inputs:
        logger.warning(f"No potential source video files found in {temp_dir_path} for project {project_id}.")
        return None

    # Assume the first found file is the source video
    # Consider adding more robust logic if multiple matches are possible
    local_input_path = os.path.join(temp_dir_path, possible_inputs[0])
    if not os.path.isfile(local_input_path):
        logger.error(f"Potential source video file found by name, but does not exist: {local_input_path}")
        return None

    logger.info(f"Found source video for project {project_id}: {local_input_path}")
    return local_input_path


def _show_export_tab(project: Dict[str, Any], ai_module: Optional[Any]) -> None:
    """Displays export options and handles the clip export process."""
    st.subheader("📤 Export Clips")
    clips = project.get("clips", [])
    project_base_dir = project.get("base_dir_path")
    project_id = project["id"]
    project_name = project.get("name", "project")

    # --- Locate Source Video ---
    local_input_path = _find_source_video(project_base_dir, project_id)

    if not local_input_path:
        st.error(
            "❌ Critical Error: Could not locate the original source video file in the project directory. "
            f"Looked in '{os.path.join(project_base_dir or '', 'temp')}'. Export is disabled."
        )
        return

    if not clips:
        st.info("This project has no analyzed clips to export.")
        return

    # --- Selection Controls ---
    st.markdown("#### 1. Select Clips for Export")
    export_filter_cols = st.columns(3)
    project_settings = project.get("settings", {})

    with export_filter_cols[0]:
        default_min_score = int(project_settings.get("score_threshold", 0))
        min_score_export = st.slider(
            "Minimum Initial Score", 0, 100, default_min_score,
            key=f"export_min_score_{project_id}",
            help="Only include clips with an initial score above this value."
        )
    with export_filter_cols[1]:
        has_ai_viral_scores = any(c.get("ai_viral_score") is not None for c in clips)
        min_ai_viral = st.slider(
            "Min AI Viral Score", 0, 100, 0, # Default to 0
            key=f"export_min_viral_{project_id}",
            disabled=not has_ai_viral_scores,
            help="Only include clips with an AI Viral Score above this value (requires AI Board analysis)."
        )
    with export_filter_cols[2]:
        max_clips_to_export = st.number_input(
            "Max Clips to Export", min_value=1, max_value=len(clips),
            value=min(10, len(clips)), step=1,
            key=f"export_max_clips_{project_id}",
            help="Limit the number of exported clips (highest scoring first)."
        )

    # --- Apply Filters and Selection ---
    # Filter based on score thresholds
    export_clips_filtered = [
        c for c in clips
        if c.get("score", 0) >= min_score_export and
           (not has_ai_viral_scores or c.get("ai_viral_score", -1) >= min_ai_viral)
    ]
    # Sort by score (descending) to select the top N
    export_clips_filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    # Select the top N clips
    export_clips_final = export_clips_filtered[:max_clips_to_export]

    if not export_clips_final:
        st.warning("No clips match the current selection criteria.")
        return # Stop if no clips are selected

    st.markdown(f"**Selected {len(export_clips_final)} clips for export:**")
    # Show a compact preview of selected clips
    clip_display.display_clips_gallery(
        export_clips_final,
        cols=min(len(export_clips_final), 5), # Limit columns in preview
        show_details=False
    )

    # --- Export Options ---
    st.markdown("#### 2. Configure Export Options")
    default_export_format = project_settings.get("export_format", EXPORT_FORMATS[0])
    try:
        default_format_index = EXPORT_FORMATS.index(default_export_format)
    except ValueError:
        logger.warning(f"Invalid default export format '{default_export_format}' found in project settings. Falling back.")
        default_format_index = 0 # Fallback to the first valid format

    export_opts_cols = st.columns(2)
    with export_opts_cols[0]:
        platform = st.selectbox(
            "Optimize For Platform", EXPORT_PLATFORMS, index=0,
            key=f"export_platform_{project_id}",
            help="Adjusts encoding/resolution for target platform (future feature)." # Note: Platform optimization might not be fully implemented yet
        )
    with export_opts_cols[1]:
        export_format_selection = st.selectbox(
            "Encoding Quality/Strategy", EXPORT_FORMATS, index=default_format_index,
            key=f"export_quality_format_{project_id}",
            help="web_optimized=Faster/Smaller, high_quality=Slower/Larger, copy_only=Fastest (no re-encode)."
        )

    # --- Export Action ---
    st.markdown("#### 3. Export")
    if st.button(
        "🚀 Export Selected Clips", type="primary", use_container_width=True,
        key=f"export_clips_btn_{project_id}",
    ):
        if not project_base_dir:
             st.error("Project base directory is missing. Cannot determine export location.")
             return

        export_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a unique subdirectory for this specific export batch
        export_output_dir = os.path.join(
            project_base_dir, "exports", f"{project_name.replace(' ', '_')}_{export_timestamp}"
        )
        try:
            os.makedirs(export_output_dir, exist_ok=True)
            logger.info(f"Exporting clips for project {project_id} to {export_output_dir}")
        except OSError as e:
            logger.error(f"Failed to create export directory {export_output_dir}: {e}", exc_info=True)
            st.error(f"Failed to create export directory: {e}")
            return

        st.info(f"Starting export to: {os.path.basename(export_output_dir)}")
        exported_files_info = []
        max_workers_clip_gen = int(project_settings.get("max_workers_clip_gen", 4))

        with st.spinner(f"Exporting {len(export_clips_final)} clips using up to {max_workers_clip_gen} workers..."):
            try:
                # Determine which export function to use (AI optimized or standard)
                export_func = None
                if ai_module and hasattr(ai_module, "export_clips_optimized"):
                    export_func = ai_module.export_clips_optimized
                    logger.info(f"Using AI-optimized export function for project {project_id}.")
                else:
                    export_func = generate_clips # Standard function from media_utils
                    logger.info(f"Using standard clip generation for export for project {project_id}.")

                # Prepare arguments based on the function being used
                if export_func == generate_clips:
                     segments_to_export = [(c["start"], c["end"]) for c in export_clips_final]
                     exported_files_info = export_func(
                         input_filepath=local_input_path,
                         segments=segments_to_export,
                         output_dir=export_output_dir,
                         export_format=export_format_selection, # Pass quality setting
                         max_workers=max_workers_clip_gen,
                     )
                elif hasattr(ai_module, "export_clips_optimized"): # Check again just in case
                     exported_files_info = export_func( # Call AI optimized version
                         clips=export_clips_final,
                         input_filepath=local_input_path,
                         output_dir=export_output_dir,
                         platform=platform.lower().replace(" ", "_"), # Pass platform hint
                         quality=export_format_selection, # Pass quality setting
                         max_workers=max_workers_clip_gen,
                     )
                else:
                    # This case should ideally not be reached due to checks above
                    st.error("Internal Error: Could not determine export function.")
                    logger.error("Export logic failed to select a valid export function.")


            except Exception as export_err:
                logger.error(f"Export process failed for project {project_id}: {export_err}", exc_info=True)
                st.error(f"Export process encountered an error: {export_err}")

        # --- Display Download Links ---
        if exported_files_info:
            st.success(f"Successfully exported {len(exported_files_info)} clips!")
            st.balloons()
            st.markdown("#### Download Exported Clips:")

            files_for_download = []
            for info in exported_files_info:
                # Standardize accessing the file path (might differ between export functions)
                file_path = info.get("path") or info.get("exported_path")
                if file_path and os.path.isfile(file_path):
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        files_for_download.append({
                            "path": file_path,
                            "name": os.path.basename(file_path),
                            "size_mb": file_size_mb,
                        })
                    except OSError as size_err:
                        logger.warning(f"Could not get file size for {file_path}: {size_err}")
                        # Still add file if size calculation fails
                        files_for_download.append({"path": file_path, "name": os.path.basename(file_path), "size_mb": 0})
                else:
                    logger.warning(f"Exported file path not found or invalid for a clip: {file_path}")

            if files_for_download:
                # Display download buttons, maybe in columns
                dl_cols = st.columns(min(len(files_for_download), 4)) # Max 4 columns
                for i, file_info in enumerate(files_for_download):
                    col = dl_cols[i % len(dl_cols)]
                    with col:
                        try:
                            with open(file_info["path"], "rb") as fp:
                                st.download_button(
                                    label=f"{file_info['name']} ({file_info['size_mb']:.1f} MB)",
                                    data=fp,
                                    file_name=file_info["name"],
                                    mime="video/mp4", # Assume mp4, might need more sophisticated MIME detection
                                    key=f"download_{i}_{project_id}_{export_timestamp}", # Unique key
                                )
                        except Exception as e:
                            logger.error(f"Error preparing download button for {file_info['name']}: {e}", exc_info=True)
                            st.error(f"Could not prepare '{file_info['name']}' for download.")
            else:
                st.warning("Export completed, but no valid files were found for download. Check logs.")
        else:
            st.error("Clip export finished, but no files were generated successfully. Check logs for errors.")


def _show_settings_tab(project: Dict[str, Any]) -> None:
    """Displays project settings, allows name editing, and project deletion."""
    st.subheader("🔧 Project Settings")
    project_id = project["id"]
    project_dir = project.get("base_dir_path")
    project_name = project.get("name", "")

    # --- Form for Editable Settings (Currently only Name) ---
    with st.form(key=f"project_settings_form_{project_id}"):
        st.markdown("**Basic Information**")
        new_name = st.text_input(
            "Project Name", value=project_name, key=f"proj_name_{project_id}"
        )

        info_cols = st.columns(2)
        with info_cols[0]:
            st.text_input("Project ID", value=project_id, disabled=True)
            created_at_str = project.get("created_at", "N/A")
            if created_at_str != "N/A":
                try:
                    # Attempt to parse and format ISO date string
                    created_at_dt = datetime.datetime.fromisoformat(created_at_str)
                    created_at_display = created_at_dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    created_at_display = created_at_str # Show raw if parsing fails
            else:
                created_at_display = "N/A"
            st.text_input("Created At", value=created_at_display, disabled=True)

        with info_cols[1]:
            st.text_input("Source Type", value=project.get("source_type", "N/A").capitalize(), disabled=True)
            source_path = project.get("source_path", "N/A")
            # Truncate long paths/URLs for display in text input
            source_path_display = (
                f"...{source_path[-47:]}" if len(source_path) > 50 else source_path
            )
            st.text_input("Source Path/URL", value=source_path_display, disabled=True, help=f"Full source: {source_path}") # Show full path in help tooltip
            st.caption(f"Project Files Directory: `{project_dir or 'N/A'}`")

        st.markdown("---")
        st.markdown("**Analysis Configuration Used** (Read-only)")
        st.caption("These settings were used when the project was created and cannot be changed here.")
        current_settings = project.get("settings", {})
        settings_cols = st.columns(3)
        # Helper to display setting value or 'N/A'
        def get_setting(key: str, default: Any = "N/A") -> str:
            return str(current_settings.get(key, default))

        with settings_cols[0]:
            st.text_input("Segment Length (s)", value=get_setting("chunk_size"), disabled=True)
            st.text_input("Frame Sample Rate (s)", value=get_setting("frame_sample_rate"), disabled=True)
            st.text_input("Score Threshold", value=get_setting("score_threshold"), disabled=True)
        with settings_cols[1]:
            st.text_input("Initial AI Provider", value=get_setting("ai_provider").capitalize(), disabled=True)
            st.text_input("Initial AI Model", value=get_setting("ai_model"), disabled=True)
            st.text_input("Audio Analyzed", value=get_setting("analyze_audio", False), disabled=True)
        with settings_cols[2]:
            st.text_input("Clip Encoding Format", value=get_setting("export_format"), disabled=True)
            st.text_input("AI Image Quality (%)", value=get_setting("compression_quality"), disabled=True)
            st.text_input("AI Image Max Resolution", value=get_setting("max_resolution"), disabled=True)

        submitted = st.form_submit_button("💾 Save Project Name")
        if submitted:
            cleaned_new_name = new_name.strip()
            if cleaned_new_name and cleaned_new_name != project_name:
                conn = None
                try:
                    # Connect to the database and update the name
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE projects SET name = ? WHERE id = ?",
                        (cleaned_new_name, project_id),
                    )
                    conn.commit()
                    st.success("Project name updated successfully!")
                    logger.info(f"Project {project_id} name updated to '{cleaned_new_name}'")
                    # Rerun to reflect name change in title and reload project data implicitly
                    st.rerun()
                except sqlite3.Error as e:
                    logger.error(
                        f"Failed to update project {project_id} name in database: {e}",
                        exc_info=True,
                    )
                    st.error(f"Failed to update project name in database: {e}")
                finally:
                    if conn:
                        conn.close()
            elif not cleaned_new_name:
                st.warning("Project name cannot be empty.")
            else:
                st.info("No changes made to the project name.")

    st.divider()

    # --- Danger Zone: Project Deletion ---
    st.markdown("### Danger Zone")
    st.warning("Deleting a project is permanent and removes all associated files and data.")

    # Use session state to manage the two-step delete confirmation
    confirm_key = f"confirming_delete_{project_id}"

    if st.session_state.get(confirm_key):
        st.error(f"🔥 ARE YOU ABSOLUTELY SURE you want to delete project '{project_name}'?")
        st.caption(f"Project files at `{project_dir or 'Unknown Location'}` will be removed.")
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if st.button("✅ Yes, DELETE Project", key=f"confirm_delete_yes_{project_id}", type="primary"):
                logger.warning(f"User initiated permanent deletion for project {project_id} ('{project_name}').")
                deleted_db = False
                try:
                    # Delete from database first
                    deleted_db = delete_project(project_id)
                    if deleted_db:
                        st.success(f"Project '{project_name}' removed from database.")
                        logger.info(f"Project {project_id} deleted from database.")
                        # Clean up files only if DB deletion was successful
                        if project_dir:
                             cleanup_project_directories(project_dir) # Logs internally
                             st.info(f"Attempted cleanup of project files at {project_dir}")
                        else:
                             st.warning("Project directory path not found, cannot clean up files.")
                    else:
                        st.error("Failed to delete project from database. Files were not removed.")
                        logger.error(f"Failed database deletion for project {project_id}.")

                except Exception as del_err:
                     logger.error(f"Error during project deletion process for {project_id}: {del_err}", exc_info=True)
                     st.error(f"An error occurred during deletion: {del_err}")

                # Reset confirmation state and navigate away regardless of success/failure
                st.session_state[confirm_key] = False
                st.session_state.current_project_id = None # Clear current project view
                st.session_state.app_mode = "📁 Projects" # Go back to list
                time.sleep(1) # Brief pause to show message
                st.rerun()

        with confirm_cols[1]:
            if st.button("❌ Cancel Deletion", key=f"confirm_delete_no_{project_id}"):
                logger.info(f"User cancelled deletion for project {project_id}.")
                st.session_state[confirm_key] = False # Reset confirmation flag
                st.rerun() # Rerun to hide confirmation UI

    else:
        # Show the initial delete button
        if st.button("❌ Delete This Project", key=f"delete_project_btn_{project_id}"):
            st.session_state[confirm_key] = True # Set flag to trigger confirmation on next rerun
            st.rerun()


def _show_ai_board_tab(project: Dict[str, Any], ai_module: Optional[Any]) -> None:
    """Displays the AI Board analysis interface using the AI module."""
    st.subheader("🧠 AI Board of Directors Analysis")
    st.caption("Engage a virtual board of AI experts to review and score your clips.")

    # Check prerequisites for the AI Board
    if not ai_module or not hasattr(ai_module, "display_ai_board_ui"):
        st.info(
            "AI Board functionality requires an initialized AI module with the "
            "'display_ai_board_ui' method. Check application setup."
        )
        return

    # Configuration for the board is typically managed elsewhere (e.g., sidebar or app settings)
    # We expect it to be in session_state or passed via ai_module settings
    board_config = st.session_state.get("selected_board_config", {}) # Example key

    if not board_config.get("board_enabled", False):
        st.info("AI Board is disabled. Enable and configure it via the application settings/sidebar.")
        return
    if not board_config.get("board_members"):
        st.info("No AI Board members are configured. Select members in the settings/sidebar.")
        return
    # A chairperson might be optional depending on the implementation
    # if not board_config.get("chairperson"):
    #     st.warning("No AI Board chairperson selected. Synthesis might be limited.")

    clips = project.get("clips", [])
    if not clips:
        st.info("This project has no clips to analyze with the AI Board.")
        return

    # Delegate the UI rendering and logic to the AI module's method
    try:
        # The AI module's function should handle its own state, API calls, and display
        ai_module.display_ai_board_ui(project, board_config)
    except Exception as e:
        logger.error(f"Error displaying/running AI Board UI for project {project['id']}: {e}", exc_info=True)
        st.error(f"An error occurred while running the AI Board: {e}")


# --- Main Page Function ---


def show(project_id: str) -> None:
    """
    Displays the main detail page for a specific project, organizing
    information into tabs.
    """
    # Attempt to get the AI Module instance from session state
    ai_module = st.session_state.get("ai_module")

    # Load the full project data, including clips and settings
    project = load_project(project_id)

    if not project:
        st.error(f"Project with ID '{project_id}' not found or could not be loaded!")
        logger.error(f"Failed attempt to load project details for ID: {project_id}")
        if st.button("◀️ Back to Projects List"):
            st.session_state.current_project_id = None
            st.session_state.app_mode = "📁 Projects"
            st.rerun()
        return # Stop execution if project loading failed

    st.title(f"🎬 Project: {project.get('name', 'Unnamed Project')}")

    # --- Header Metrics ---
    header_cols = st.columns(4)
    clips = project.get("clips", [])
    clip_count = len(clips)
    video_info = project.get("video_info", {})
    video_duration = video_info.get("duration")
    avg_score = (
        sum(c.get("score", 0) for c in clips) / clip_count if clip_count > 0 else 0
    )
    highest_score = max((c.get("score", 0) for c in clips), default=0)

    header_cols[0].metric("Clips Identified", clip_count)
    header_cols[1].metric(
        "Video Duration", f"{video_duration:.1f}s" if video_duration else "N/A"
    )
    header_cols[2].metric("Avg. Initial Score", f"{avg_score:.1f}" if clips else "N/A")
    header_cols[3].metric("Highest Initial Score", f"{highest_score}" if clips else "N/A")

    st.divider()

    # --- Tabs Definition ---
    tab_titles = ["Clips", "Analytics"]
    # Conditionally add AI Board tab
    ai_board_available = ai_module and hasattr(ai_module, "display_ai_board_ui")
    if ai_board_available:
        tab_titles.append("🧠 AI Board")
    tab_titles.extend(["Export", "Settings"])

    # Create tabs
    tabs = st.tabs(tab_titles)
    # Map titles to tab objects for easier access
    tab_map = {title: tab for title, tab in zip(tab_titles, tabs)}

    # --- Populate Tabs ---
    # Use try-except blocks for each tab's content function to isolate errors
    with tab_map["Clips"]:
        try:
            _show_clips_tab(project)
        except Exception as e:
            logger.error(f"Error rendering Clips tab for project {project_id}: {e}", exc_info=True)
            st.error(f"An error occurred in the Clips tab: {e}")

    with tab_map["Analytics"]:
        try:
            _show_analytics_tab(project, ai_module)
        except Exception as e:
            logger.error(f"Error rendering Analytics tab for project {project_id}: {e}", exc_info=True)
            st.error(f"An error occurred in the Analytics tab: {e}")

    if ai_board_available:
        with tab_map["🧠 AI Board"]:
            try:
                _show_ai_board_tab(project, ai_module)
            except Exception as e:
                logger.error(f"Error rendering AI Board tab for project {project_id}: {e}", exc_info=True)
                st.error(f"An error occurred in the AI Board tab: {e}")

    with tab_map["Export"]:
        try:
            _show_export_tab(project, ai_module)
        except Exception as e:
            logger.error(f"Error rendering Export tab for project {project_id}: {e}", exc_info=True)
            st.error(f"An error occurred in the Export tab: {e}")

    with tab_map["Settings"]:
        try:
            _show_settings_tab(project) # Pass only project, ai_module not needed here currently
        except Exception as e:
            logger.error(f"Error rendering Settings tab for project {project_id}: {e}", exc_info=True)
            st.error(f"An error occurred in the Settings tab: {e}")

    st.divider()
    # Consistent "Back" button at the bottom of the page
    if st.button("◀️ Back to Projects List", key="back_button_project_detail_bottom"):
        st.session_state.current_project_id = None # Clear the currently viewed project ID
        st.session_state.app_mode = "📁 Projects" # Set mode to navigate back
        st.rerun()