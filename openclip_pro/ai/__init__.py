
# This file contains the initialization logic for the AI module.
import logging
import time

import streamlit as st

# Set up logger for this module
logger = logging.getLogger(__name__)


def initialize_ai_integration():
    """
    Initializes the AI integration module and handles dependency availability.

    Checks if the AI module is already initialized and cached in the Streamlit
    session state. If not, it attempts to import and instantiate the main
    AIAnalysisModule. It caches the successful instance in the session state
    for reuse across Streamlit reruns. Handles potential ImportError if core
    dependencies are missing and other exceptions during initialization.

    Returns:
        AIAnalysisModule: An instance of the AI module if initialization is successful.
        None: If initialization fails due to missing dependencies or other errors.
    """
    # Reuse cached instance if available in session state
    if "ai_module" in st.session_state and st.session_state.ai_module:
        logger.debug("Using cached AI module from session state.")
        return st.session_state.ai_module

    # Log the start of the initialization process
    logger.info("Initializing AI integration...")
    start_time = time.time()

    # Ensure the "API Keys" mode is available in the navigation/UI.
    # This is added early, even if initialization fails later, allowing users
    # to potentially access API key settings.
    if "ai_integration_modes" not in st.session_state:
        st.session_state.ai_integration_modes = ["🔑 API Keys"]
        logger.debug("Added 'API Keys' to ai_integration_modes in session state.")

    try:
        # Dynamically import the main AI module class.
        # This approach contains the dependency check implicitly: if the import
        # fails, it means the core AI library structure is missing.
        # Optional provider dependencies (like specific SDKs) are typically
        # checked internally by AIAnalysisModule or its components upon use.
        from .ai_models import AIAnalysisModule

        # Instantiate the main AI module
        ai_module = AIAnalysisModule()
        logger.debug("AIAnalysisModule instance created successfully.")

        # Cache the successfully initialized module instance in session state
        st.session_state.ai_module = ai_module
        # Also cache a reference to the model registry for potential direct access
        st.session_state.model_registry = ai_module.model_registry
        logger.debug("AI module and model registry cached in session state.")

        elapsed_time = time.time() - start_time
        logger.info(
            f"AI integration initialized successfully in {elapsed_time:.2f}s"
        )
        return ai_module

    except ImportError as import_error:
        # Handle error when the core AI module or its essential imports fail
        error_message = f"Required AI dependency missing: {import_error}"
        logger.error(error_message, exc_info=True)
        st.error(
            f"Fatal Error: Missing required AI dependency ({import_error}). "
            "Please ensure all necessary packages are installed."
        )
        # Ensure ai_module is explicitly None in session state on failure
        st.session_state.ai_module = None
        return None

    except Exception as general_error:
        # Handle any other unexpected errors during initialization
        error_message = f"Failed to initialize AI module: {general_error}"
        logger.error(error_message, exc_info=True)
        st.error(
            f"Fatal Error: Failed to initialize AI integration ({general_error}). "
            "Check application logs for more details."
        )
        # Ensure ai_module is explicitly None in session state on failure
        st.session_state.ai_module = None
        return None
