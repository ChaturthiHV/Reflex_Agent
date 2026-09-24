"""Central configuration for ReflexAgent.

Loads environment variables, defines loop thresholds, and registers the
domain tasks the orchestrator can run. Adding a new use case only requires
adding an entry to DOMAIN_TASKS.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM configuration ---------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b")
GROQ_MODEL_SECONDARY = os.getenv("GROQ_MODEL_SECONDARY", "openai/gpt-oss-20b")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# --- Loop / safety configuration -----------------------------------------
MAX_ATTEMPTS = int(os.getenv("REFLEXAGENT_MAX_ATTEMPTS", "4"))
CONFIDENCE_FLOOR = float(os.getenv("REFLEXAGENT_CONFIDENCE_FLOOR", "0.4"))
REQUEST_TIMEOUT_SECONDS = 20
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.5

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LESSONS_PATH = os.path.join(BASE_DIR, "memory", "lessons.json")
CACHED_PAPERS_PATH = os.path.join(BASE_DIR, "data", "cached_papers.json")

# --- Supported languages for the voice/text interface ----------------------
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}


def _crop_disease_dataset():
    """Synthetic 'leaf symptom features -> disease class' dataset.

    Stands in for a real field dataset so the loop can be demoed end-to-end
    without needing an image dataset on stage. Swap this for a real loader
    (e.g. leaf-image embeddings) without touching any other file.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.1,
        flip_y=0.05,
        random_state=7,
    )
    return X, y


# --- Domain task registry ---------------------------------------------------
# Only "goal", "dataset_loader", "target_metric" and "target_value" change
# per domain. Everything else in the pipeline is shared.
DOMAIN_TASKS = {
    "agriculture": {
        "goal": (
            "Diagnose the crop disease from leaf symptom readings and reach "
            "reliable classification accuracy so a farmer can trust the result."
        ),
        "dataset_loader": _crop_disease_dataset,
        "target_metric": "accuracy",
        "target_value": 0.85,
        "literature_query": "plant disease classification machine learning",
    },
    # Additional domains (healthcare screening, education, financial
    # inclusion, ...) follow the same shape and can be added here without
    # changing the orchestrator or any agent.
}
