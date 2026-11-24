"""Central configuration: every path, column name and constant lives here.

Keeping them in one file means no other module hard-codes a path or a column
name, so the same code can be pointed at a renamed dataset in one place.
"""
from pathlib import Path

RANDOM_STATE = 42
# Model families tuned with RandomizedSearchCV on BOTH splits (same models => comparable bars)
TUNE_MODELS = ["RandomForest", "XGBoost"]

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"
NOTEBOOK_DIR = ROOT / "notebooks"

GLOBAL_CSV = DATA_DIR / "yield_df.csv"
INDIA_CSV = DATA_DIR / "india_crop_production.csv"

GLOBAL_MODEL_PATH = MODELS_DIR / "best_model.joblib"
INDIA_MODEL_PATH = MODELS_DIR / "india_model.joblib"
GLOBAL_META_PATH = MODELS_DIR / "global_meta.joblib"   # dataset quantiles, choices
INDIA_META_PATH = MODELS_DIR / "india_meta.joblib"

GLOBAL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
INDIA_COMPARISON_CSV = RESULTS_DIR / "india_model_comparison.csv"

# ------------------------------------------------- global dataset columns
G_INDEX = "Unnamed: 0"            # junk index column to drop
G_COUNTRY = "Area"
G_CROP = "Item"
G_YEAR = "Year"
G_TARGET = "hg/ha_yield"
G_RAIN = "average_rain_fall_mm_per_year"
G_PEST = "pesticides_tonnes"
G_TEMP = "avg_temp"
G_CAT = [G_COUNTRY, G_CROP]
G_NUM = [G_YEAR, G_RAIN, G_TEMP, G_PEST]
G_FEATURES = G_CAT + G_NUM
G_TEST_YEARS = 4                  # time split: most recent 4 years are the test set

# ---------------------------------------------------- India dataset columns
I_CROP = "Crop"
I_YEAR = "Crop_Year"
I_SEASON = "Season"
I_STATE = "State"
I_AREA = "Area"
I_PROD = "Production"
I_RAIN = "Annual_Rainfall"
I_FERT = "Fertilizer"
I_PEST = "Pesticide"
I_TARGET = "Yield"
# Fertilizer / Pesticide in the CSV are TOTALS for the whole cultivated area, so they mostly
# measure farm size. We convert them to intensity per hectare (total / Area) and use those.
I_FERT_HA = "Fertilizer_per_ha"
I_PEST_HA = "Pesticide_per_ha"
I_CAT = [I_CROP, I_SEASON, I_STATE]
I_NUM = [I_YEAR, I_AREA, I_RAIN, I_FERT_HA, I_PEST_HA]
I_FEATURES = I_CAT + I_NUM       # NOTE: Production deliberately excluded (target leak)
I_TEST_YEARS = 4
# Coconut is recorded in NUMBER OF NUTS per hectare (yields in the thousands), not tonnes,
# so it is not comparable with other crops and would dominate squared-error metrics.
I_EXCLUDE_CROPS = ["Coconut"]

HG_PER_HA_TO_T_PER_HA = 1e-4          # 1 hg = 0.1 kg and 1 t = 1000 kg, so 1 hg/ha = 1e-4 t/ha

# ------------------------------------------------ human-readable labels (XAI)
G_LABELS = {G_COUNTRY: "Country", G_CROP: "Crop", G_YEAR: "Year",
            G_RAIN: "Rainfall", G_TEMP: "Temperature", G_PEST: "Pesticide use"}
G_UNITS = {G_RAIN: "mm/yr", G_TEMP: "°C", G_PEST: "t", G_YEAR: ""}
I_LABELS = {I_CROP: "Crop", I_SEASON: "Season", I_STATE: "State", I_YEAR: "Year",
            I_AREA: "Cultivated area", I_RAIN: "Rainfall", I_FERT_HA: "Fertilizer use",
            I_PEST_HA: "Pesticide use"}
I_UNITS = {I_RAIN: "mm/yr", I_AREA: "ha", I_FERT_HA: "kg/ha", I_PEST_HA: "kg/ha", I_YEAR: ""}
