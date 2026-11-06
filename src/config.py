"""Settings in one place."""

# Short name -> model on Hugging Face. All are public (no login, no acceptance needed).
MODELS = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
}
DEFAULT_MODELS = ["xlmr", "mbert", "muril"]

BATCH_SIZE = 64
MAX_LENGTH = 64
N_BOOTSTRAP = 1000
SEED = 42

DATA_DIR = "data"
RESULTS_DIR = "results"
