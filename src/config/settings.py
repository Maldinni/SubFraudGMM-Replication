import os
from pathlib import Path
from dotenv import load_dotenv # Recebe as variáveis de ambiente

# Raiz do projeto (src/config/settings.py -> config -> src -> raiz).
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "keys.env")

# Por padrão usa a raiz do projeto (onde estão data/ e results/). Pode ser sobrescrito
# definindo LOCAL_BASEPATH no keys.env — útil se os dados ficarem fora do repositório.
LOCAL_BASEPATH = Path(os.getenv("LOCAL_BASEPATH", BASE_DIR))

OUTPUT_DIR = LOCAL_BASEPATH / "data"
PROCESSED_DIR = OUTPUT_DIR / "processed"
INTERMEDIARY_ESC_CSV_INPUT_FILE = OUTPUT_DIR / "escavadeira_final.csv"
INTERMEDIARY_MOT_CSV_INPUT_FILE = OUTPUT_DIR / "motoniveladora_final.csv"
INTERMEDIARY_ROL_CSV_INPUT_FILE = OUTPUT_DIR / "rolo_compactador_final.csv"
INTERMEDIARY_TRA_CSV_INPUT_FILE = OUTPUT_DIR / "trator_esteira_final.csv"
FINAL_GMM_CSV_INPUT_FILE = LOCAL_BASEPATH / "results" / "SubFraudGMM.csv"

TIMEOUT = 60
