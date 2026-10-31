
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import unicodedata
import sqlite3
import hashlib
import csv
import json
import re
import gc

# ============================================================
# Q4 | REVIEW EXTRACTION PIPELINE V2.4
# Python Rules + MiniLM + SQLite Embedding Cache
# ============================================================

VERSION = "2.4"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

SOURCE_FILE = (
    PROJECT_DIR / "data" / "olist"
    / "olist_order_reviews_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "output" / "preprocessing"
RULE_OUTPUT_DIR = BASE_DIR / "output" / "rule_engine"
CACHE_DIR = BASE_DIR / "cache"

for directory in [OUTPUT_DIR, RULE_OUTPUT_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# SQLite database containing persistent MiniLM vectors.
CACHE_DB = CACHE_DIR / "minilm_embeddings.sqlite3"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Change this tag if the actual model weights are replaced.
MODEL_CACHE_TAG = "v1"

MODEL_KEY = f"{MODEL_NAME}|{MODEL_CACHE_TAG}|384"

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

BATCH_SIZE = 64
DEVICE = "cpu"
EMBEDDING_DIM = 384

SCORE_THRESHOLD = 3

# Initial cosine similarity thresholds.
# These are not calibrated probabilities.
CATEGORY_THRESHOLDS = {
    "delivery_late": 0.58,
    "wrong_product": 0.60,
    "quality_defect": 0.60,
    "missing_item": 0.58,
    "seller_communication": 0.61,
    "packaging": 0.68
}

MIN_TOP_MARGIN = 0.04
SECONDARY_MARGIN = 0.035

CATEGORIES = [
    "delivery_late",
    "wrong_product",
    "quality_defect",
    "missing_item",
    "seller_communication",
    "packaging",
    "other"
]

ROUTES = [
    "python_simple",
    "llm_enrich",
    "llm_full"
]


# ============================================================
# 1. TEXT NORMALIZATION AND FINGERPRINT
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_data():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    df = pd.read_csv(
        SOURCE_FILE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quotechar='"',
        doublequote=True,
        on_bad_lines="error",
        low_memory=False
    )

    required = [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_message"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Source records: {len(df):,}")

    return df


# ============================================================
# 3. NEGATIVE REVIEW FILTERING
# ============================================================

def filter_reviews(df):

    scores = pd.to_numeric(
        df["review_score"],
        errors="coerce"
    )

    if scores.isna().any():
        raise ValueError(
            "Invalid review_score detected."
        )

    if not scores.between(1, 5).all():
        raise ValueError(
            "Unexpected review_score detected."
        )

    negative = df.loc[
        scores <= SCORE_THRESHOLD
    ].copy()

    print(f"Negative reviews: {len(negative):,}")

    return negative


# ============================================================
# 4. DATA PREPROCESSING AND DEDUPLICATION
# ============================================================

def prepare_reviews(df):

    if df["review_id"].eq("").any():
        raise ValueError(
            "Empty review_id detected."
        )

    duplicate_mask = df["review_id"].duplicated(
        keep=False
    )

    duplicate_records = df.loc[duplicate_mask]

    exact_duplicate_rows = int(
        df.duplicated(keep=False).sum()
    )

    print("\n========== DATA QUALITY ==========")

    print(
        f"Duplicate review IDs : "
        f"{duplicate_records['review_id'].nunique():,}"
    )

    print(
        f"Affected records     : "
        f"{len(duplicate_records):,}"
    )

    print(
        f"Exact duplicate rows : "
        f"{exact_duplicate_rows:,}"
    )

    all_reviews = df.copy()

    all_reviews["original_text"] = (
        all_reviews["review_comment_message"]
    )

    all_reviews["normalized_text"] = (
        all_reviews["original_text"].apply(
            normalize_text
        )
    )

    all_reviews["is_empty"] = (
        all_reviews["original_text"].str.strip() == ""
    )

    valid = all_reviews.loc[
        ~all_reviews["is_empty"]
    ].copy()

    empty = all_reviews.loc[
        all_reviews["is_empty"]
    ].copy()

    group_codes, _ = pd.factorize(
        valid["normalized_text"],
        sort=True
    )

    valid["duplicate_group_id"] = (
        "G" + (group_codes + 1).astype(str)
    )

    valid["text_frequency"] = (
        valid.groupby("normalized_text")[
            "review_id"
        ].transform("count")
    )

    unique = valid.drop_duplicates(
        subset=["normalized_text"],
        keep="first"
    ).copy()

    unique["text_hash"] = (
        unique["normalized_text"].apply(text_hash)
    )

    print(
        f"Valid comments       : {len(valid):,}"
    )

    print(
        f"Empty comments       : {len(empty):,}"
    )

    print(
        f"Unique texts         : {len(unique):,}"
    )

    print(
        f"Duplicate text rows  : "
        f"{len(valid) - len(unique):,}"
    )

    return all_reviews, valid, empty, unique


# ============================================================
# 5. EXPORT AND INTEGRITY VALIDATION
# ============================================================

def save_csv(df, path):

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        lineterminator="\n"
    )

    return path


def verify_csv(path, original, columns):

    check = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error"
    )

    if len(check) != len(original):
        raise ValueError(
            f"Row count mismatch: {path.name}"
        )

    for col in columns:

        if check[col].tolist() != (
            original[col].astype(str).tolist()
        ):
            raise ValueError(
                f"Data integrity mismatch: "
                f"{path.name}, column={col}"
            )

    return True


def export_preprocessing(
    all_reviews,
    valid,
    empty,
    unique
):

    datasets = {
        "all": all_reviews,
        "text": valid,
        "empty": empty,
        "unique_texts": unique
    }

    files = {}

    for name, data in datasets.items():

        path = (
            OUTPUT_DIR
            / f"negative_reviews_{name}.csv"
        )

        save_csv(data, path)
        files[name] = path

    mapping = valid[
        [
            "review_id",
            "order_id",
            "normalized_text",
            "duplicate_group_id",
            "text_frequency"
        ]
    ].copy()

    mapping_path = (
        OUTPUT_DIR / "review_id_mapping.csv"
    )

    save_csv(mapping, mapping_path)
    files["mapping"] = mapping_path

    verify_csv(
        files["text"],
        valid,
        ["review_id", "original_text"]
    )

    verify_csv(
        files["unique_texts"],
        unique,
        ["review_id", "original_text"]
    )

    verify_csv(
        files["mapping"],
        mapping,
        ["review_id", "normalized_text"]
    )

    print("CSV integrity verification: PASS")

    return files


# ============================================================
# 6. PYTHON BUSINESS RULES
# ============================================================

ISSUE_RULES = {

    # Delivery delay and explicitly undelivered orders.
    "delivery_late": [

        r"\bentrega atrasada\b",
        r"\batraso na entrega\b",
        r"\bdemora para entregar\b",
        r"\bdemora pra entregar\b",
        r"\bdemorou muito para entregar\b",
        r"\bentrega muito demorada\b",
        r"\bpassou do prazo de entrega\b",
        r"\bfora do prazo de entrega\b",
        r"\bnao cumpriu o prazo de entrega\b",
        r"\bnao foi cumprido o prazo\b",
        r"\bproduto chegou atrasado\b",
        r"\bpedido chegou atrasado\b",
        r"\bentregue com atraso\b",
        r"\bchegou depois do prazo\b",
        r"\bpoderia ser mais rapida a entrega\b",
        r"\bentrega poderia ser mais rapida\b",
        r"\batrasou a entrega\b",
        r"\bnao respeitou o prazo de entrega\b",
        r"\b.{0,50}ate agora nao chegou\b",
        r"\b.{0,50}ate hoje nao chegou\b",

        # Explicit expressions of delay.
        r"\btirando a demora\b",
        r"\bapesar da demora\b",
        r"\bpor causa da demora\b",
        r"\bpor demora.{0,60}(?:cancelamento|cancelado|pedido)\b",
        r"\bcancelamento.{0,60}por demora\b",
        r"\bproduto com \d+ dias de atraso\b",
        r"\bpedido com \d+ dias de atraso\b",

        # Product explicitly not received.
        r"\bnao recebi o produto\b",
        r"\bnao recebi meu produto\b",
        r"\bnao recebi a mercadoria\b",
        r"\bproduto nao foi entregue\b",
        r"\bpedido nao foi entregue\b",
        r"\bproduto ainda nao chegou\b",
        r"\bpedido ainda nao chegou\b",
        r"\bnao chegou ate hoje\b",
        r"\bnao chegou ate agora\b",
        r"\bproduto nao chegou\b",
        r"\bpedido nao chegou\b",
        r"\bnao entregaram o pedido\b",
        r"\bnao entregaram o produto\b",
        r"\bmercadoria nao foi entregue\b",
        r"\bproduto continua sem entrega\b",
        r"\bpedido nunca chegou\b",
        r"\bcontinuo aguardando (?:o produto|a encomenda|o pedido)\b",
        r"\bainda nao recebi\b",
        r"\bate agora nao recebi\b",
        r"\bate hoje nao recebi\b",
        r"\bnao recebi e ate agora\b",

        # Tracking/dispatch delay.
        r"\brastreio falho\b",
        r"\bproduto ficou parado na postagem\b",
        r"\bpedido ficou parado na postagem\b",
        r"\bpedido parado no transporte\b",
        r"\bproduto parado no transporte\b"
    ],

    # Wrong product / mismatch with order or advertisement.
    "wrong_product": [

        r"\brecebi o produto errado\b",
        r"\brecebi um produto diferente\b",
        r"\bveio outro produto\b",
        r"\benviaram (?:o produto|um produto) errado\b",
        r"\bdiferente do que (?:comprei|pedi)\b",
        r"\bnao corresponde ao anunciado\b",
        r"\bmodelo diferente\b",
        r"\btamanho diferente do anunciado\b",
        r"\bcor diferente da anunciada\b",
        r"\bnao veio o mesmo produto\b",
        r"\bproduto diferente do anunciado\b",
        r"\bproduto nao e o mesmo da foto\b",
        r"\bnao e o produto que comprei\b",
        r"\bmandaram o produto errado\b",
        r"\bproduto veio diferente da foto\b",
        r"\brecebi outro modelo\b",
        r"\bproduto entregue nao foi o mesmo\b",
        r"\bnao foi o mesmo do anuncio\b",
        r"\bdiferente do anuncio\b",
        r"\bdiferente da foto do anuncio\b",
        r"\bproduto nao corresponde a foto\b",
        r"\bproduto enviado diferente\b",
        r"\btrocaram o produto por outro\b",
        r"\bproduto foi substituido por outro\b",
        r"\bcomprei um recebi outro\b"
    ],

    # Defects in the product itself.
    "quality_defect": [

        r"\bproduto com defeito\b",
        r"\bveio com defeito\b",
        r"\bapresentou defeito\b",
        r"\bnao funciona\b",
        r"\bnao esta funcionando\b",
        r"\bparou de funcionar\b",
        r"\bdeixou de funcionar\b",
        r"\bproduto quebrado\b",
        r"\bproduto danificado\b",
        r"\bproduto chegou quebrado\b",
        r"\bproduto chegou danificado\b",
        r"\bveio quebrado\b",
        r"\bveio danificado\b",
        r"\bnao liga\b",
        r"\bnao carrega\b",
        r"\bproduto veio com problema\b",
        r"\bproduto apresentou problema\b",
        r"\bnao funciona corretamente\b",
        r"\bpeca quebrada\b",
        r"\bproduto nao funciona\b",
        r"\bproduto de pessima qualidade\b",
        r"\bbaixa qualidade\b",
        r"\bqualidade muito ruim\b",
        r"\bacabamento ruim\b",
        r"\bmaterial de pessima qualidade\b",
        r"\bestourou\b",
        r"\bexplodiu\b",
        r"\bveio rachado\b",
        r"\bproduto rachado\b",
        r"\bproduto veio amassado\b",
        r"\bum dos controles nao esta funcionando\b",
        r"\bproduto veio rasgado\b",
        r"\btampa.{0,30}(?:rachada|quebrada)\b"
        r"\bproduto.{0,40}\brasgado\b",
        r"\b(?:carregador|controle|aparelho).{0,20}\bquebrad[oa]\b",
    ],

    # Missing products, quantities, accessories or parts.
    "missing_item": [

        r"\bfaltou (?:um |uma |o |a |os |as )",
        r"\bfaltaram (?:as |os )?(?:pecas|itens|acessorios)\b",
        r"\bveio faltando\b",
        r"\best[aá] faltando\b",
        r"\bestao faltando\b",
        r"\bnao veio o acessorio\b",
        r"\bnao vieram os acessorios\b",
        r"\brecebi apenas um\b",
        r"\brecebi apenas uma\b",
        r"\brecebi somente um\b",
        r"\brecebi somente uma\b",
        r"\bparte do pedido nao veio\b",
        r"\bveio apenas uma unidade\b",
        r"\bfaltou uma unidade\b",
        r"\bfaltou uma peca\b",
        r"\bfaltou um item\b",
        r"\bpedido incompleto\b",
        r"\bproduto incompleto\b",
        r"\bnao veio tudo\b",
        r"\bnao recebi todos os itens\b",
        r"\bveio somente uma unidade\b",
        r"\bentregaram somente um\b",
        r"\bentregaram somente uma\b",
        r"\bentregaram apenas um\b",
        r"\bentregaram apenas uma\b",
        r"\bso veio um\b",
        r"\bso veio uma\b",
        r"\bso recebi um\b",
        r"\bso recebi uma\b",
        r"\bquantidade recebida menor\b",
        r"\bveio menos do que comprei\b",
        r"\brecebi menos unidades\b",
        r"\bnao recebi a quantidade completa\b",
        r"\bveio faltando acessorios\b",
        r"\bfaltando parafusos\b"
    ],

    # Seller communication and customer service.
    "seller_communication": [

        r"\bnao consigo contato\b",
        r"\bdificil contato com\b",
        r"\bnao respondem\b",
        r"\bnao responde(?:m)? (?:ao|aos|as) emails?\b",
        r"\bnao atendem o telefone\b",
        r"\bnao tem atendimento online\b",
        r"\bnao consigo falar com\b",
        r"\bsem resposta do vendedor\b",
        r"\bsem retorno do vendedor\b",
        r"\bpessimo atendimento\b",
        r"\bnao respondem minhas mensagens\b",
        r"\bvendedor nao responde\b",
        r"\bloja nao responde\b",
        r"\bnao obtive resposta\b",
        r"\bnao tive retorno\b",
        r"\bnao consigo atendimento\b",
        r"\btelefone nao atende\b",
        r"\bnao recebi resposta\b",
        r"\bdificuldade de falar com\b",
        r"\bnao me deram retorno\b",
        r"\bnao tive resposta\b",
        r"\bnao obtenho resposta\b",
        r"\bnao obtenho retorno\b",
        r"\bnao obtive retorno\b",
        r"\bnao obtenho resposta retorno\b"
    ],

    # STRICT PACKAGING:
    # Only physical damage to the packaging itself.
    # MiniLM alone cannot generate this category.
    "packaging": [

        r"\bembalagem amassada\b",
        r"\bembalagem rasgada\b",
        r"\bembalagem quebrada\b",
        r"\bembalagem danificada\b",
        r"\bembalagem furada\b",
        r"\bembalagem estourada\b",
        r"\bcaixa amassada\b",
        r"\bcaixa rasgada\b",
        r"\bcaixa quebrada\b",
        r"\bcaixa danificada\b",
        r"\bcaixa furada\b",
        r"\bcaixa estourada\b",
        r"\bpacote amassado\b",
        r"\bpacote rasgado\b",
        r"\bpacote danificado\b",
        r"\bpacote furado\b",
        r"\bpacote estourado\b",
        r"\bsaco da embalagem rasgado\b",
        r"\bsacola rasgada\b",
        r"\bplastico da embalagem rasgado\b",
        r"\bembalagem chegou danificada\b",
        r"\bcaixa chegou amassada\b",
        r"\bcaixa chegou danificada\b",
        r"\bcaixa toda rasgada\b",
        r"\bcaixa estava toda amassada\b",
        r"\bcaixas rasgadas\b",
        r"\brasgos na caixa\b",
        r"\brasgada a caixa\b"
    ]
}


# ============================================================
# 7. DELIVERY STATE AND SEMANTIC GUARDS
# ============================================================

NOT_RECEIVED_PATTERNS = [

    r"\bnao recebi o produto\b",
    r"\bnao recebi meu produto\b",
    r"\bnao recebi a mercadoria\b",
    r"\bproduto nao foi entregue\b",
    r"\bpedido nao foi entregue\b",
    r"\bproduto ainda nao chegou\b",
    r"\bpedido ainda nao chegou\b",
    r"\bnao chegou ate hoje\b",
    r"\bnao chegou ate agora\b",
    r"\bproduto nao chegou\b",
    r"\bpedido nao chegou\b",
    r"\bnao entregaram o pedido\b",
    r"\bnao entregaram o produto\b",
    r"\bmercadoria nao foi entregue\b",
    r"\bpedido nunca chegou\b",
    r"\bainda nao recebi\b",
    r"\bate agora nao recebi\b",
    r"\bate hoje nao recebi\b",
    r"\bcontinuo aguardando\b"
]

DELAY_PATTERNS = [

    r"\batrasad[oa]\b",
    r"\batraso\b",
    r"\bdemora\b",
    r"\bdemorou\b",
    r"\bpassou do prazo\b",
    r"\bfora do prazo\b",
    r"\bdepois do prazo\b",
    r"\bprazo nao cumprido\b",
    r"\btirando a demora\b",
    r"\bpor demora\b"
]

POSITIVE_DELIVERY_PATTERNS = [

    r"\bprazo de entrega (?:e|foi) excelente\b",
    r"\bentrega (?:e|foi) excelente\b",
    r"\bentrega muito rapida\b",
    r"\bentrega rapida\b",
    r"\bentregaram antes do prazo\b",
    r"\bentrega antes do prazo\b",
    r"\bproduto chegou antes do prazo\b",
    r"\bproduto chegou no prazo\b",
    r"\bentrega dentro do prazo\b"
]

CANCELLATION_ONLY_PATTERN = re.compile(
    r"\b(?:pedido|compra)\b.{0,40}\b(?:cancelado|cancelada|cancelamento)\b"
    r"|\b(?:cancelado|cancelada|cancelamento)\b.{0,40}\b(?:pedido|compra)\b"
)


def has_pattern(text, patterns):

    normalized = normalize_text(text)

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def detect_delivery_state(text):

    normalized = normalize_text(text)

    not_received = has_pattern(
        normalized,
        NOT_RECEIVED_PATTERNS
    )

    delayed = has_pattern(
        normalized,
        DELAY_PATTERNS
    )

    if not_received and delayed:
        return "delayed_and_not_received"

    if not_received:
        return "not_received"

    if delayed:
        return "delayed"

    return ""


def cancellation_without_reason(text):

    normalized = normalize_text(text)

    if not CANCELLATION_ONLY_PATTERN.search(normalized):
        return False

    has_delivery_evidence = (
        has_pattern(normalized, DELAY_PATTERNS)
        or has_pattern(normalized, NOT_RECEIVED_PATTERNS)
    )

    return not has_delivery_evidence


def positive_delivery_without_delay(text):

    normalized = normalize_text(text)

    positive = has_pattern(
        normalized,
        POSITIVE_DELIVERY_PATTERNS
    )

    explicit_negative = (
        has_pattern(normalized, DELAY_PATTERNS)
        or has_pattern(normalized, NOT_RECEIVED_PATTERNS)
    )

    return positive and not explicit_negative


# ============================================================
# 8. MINILM CATEGORY PROTOTYPES
# ============================================================

CATEGORY_PROTOTYPES = {

    "delivery_late": [
        "O pedido está atrasado e chegou depois do prazo prometido.",
        "A entrega demorou mais do que o prazo informado.",
        "O produto foi entregue fora do prazo previsto.",
        "A transportadora atrasou a entrega do pedido.",
        "O prazo de entrega não foi cumprido.",
        "O cliente ainda não recebeu o produto comprado.",
        "O pedido não chegou até a data prevista."
    ],

    "wrong_product": [
        "Recebi um produto diferente daquele que comprei.",
        "O produto recebido não corresponde ao anúncio.",
        "O modelo, tamanho ou cor recebido está errado.",
        "O produto entregue é diferente da foto apresentada.",
        "O vendedor enviou um item diferente do solicitado."
    ],

    "quality_defect": [
        "O produto apresenta defeito de fabricação.",
        "O produto está quebrado e não funciona.",
        "O aparelho parou de funcionar corretamente.",
        "O produto recebido tem baixa qualidade.",
        "O item apresenta problemas de funcionamento.",
        "O produto está rachado ou com defeito."
    ],

    "missing_item": [
        "O pedido chegou incompleto e faltam alguns itens.",
        "O cliente comprou duas unidades mas recebeu apenas uma.",
        "Faltam peças ou acessórios que deveriam acompanhar o produto.",
        "O cliente recebeu apenas parte dos produtos comprados.",
        "A quantidade entregue é menor que a quantidade encomendada."
    ],

    "seller_communication": [
        "O cliente não consegue entrar em contato com o vendedor.",
        "A loja não responde às mensagens do cliente.",
        "O vendedor não oferece atendimento adequado.",
        "O cliente tentou falar com a loja mas não recebeu resposta.",
        "O atendimento ao cliente não resolve o problema."
    ],

    # Kept for diagnostic scoring only.
    # It is NEVER allowed to create a packaging candidate.
    "packaging": [
        "A embalagem externa chegou amassada.",
        "A caixa do produto chegou rasgada.",
        "O pacote recebido estava fisicamente danificado.",
        "A embalagem de transporte estava fisicamente rasgada.",
        "O saco ou caixa que continha o produto chegou rasgado."
    ]
}


# ============================================================
# 9. SQLITE EMBEDDING CACHE
# ============================================================

def open_cache():

    conn = sqlite3.connect(
        str(CACHE_DB),
        timeout=60
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_embeddings (
            model_key TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (model_key, text_hash)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prototype_embeddings (
            model_key TEXT NOT NULL,
            prototype_key TEXT NOT NULL,
            category TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (model_key, prototype_key)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache_metadata (
            model_key TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            cache_tag TEXT NOT NULL
        )
    """)

    conn.execute("""
        INSERT OR REPLACE INTO cache_metadata
        (model_key, model_name, embedding_dimension, cache_tag)
        VALUES (?, ?, ?, ?)
    """, (
        MODEL_KEY,
        MODEL_NAME,
        EMBEDDING_DIM,
        MODEL_CACHE_TAG
    ))

    conn.commit()

    return conn


def fetch_review_cache(conn):

    rows = conn.execute("""
        SELECT text_hash, dimension, embedding
        FROM review_embeddings
        WHERE model_key = ?
    """, (MODEL_KEY,)).fetchall()

    result = {}

    invalid = []

    for key, dimension, blob in rows:

        vector = np.frombuffer(
            blob,
            dtype=np.float32
        )

        if (
            dimension != EMBEDDING_DIM
            or len(vector) != EMBEDDING_DIM
            or not np.isfinite(vector).all()
        ):
            invalid.append(key)
            continue

        result[key] = vector.copy()

    if invalid:

        conn.executemany("""
            DELETE FROM review_embeddings
            WHERE model_key = ? AND text_hash = ?
        """, [
            (MODEL_KEY, key)
            for key in invalid
        ])

        conn.commit()

    return result


def fetch_prototype_cache(conn):

    rows = conn.execute("""
        SELECT prototype_key, dimension, embedding
        FROM prototype_embeddings
        WHERE model_key = ?
    """, (MODEL_KEY,)).fetchall()

    result = {}
    invalid = []

    for key, dimension, blob in rows:

        vector = np.frombuffer(
            blob,
            dtype=np.float32
        )

        if (
            dimension != EMBEDDING_DIM
            or len(vector) != EMBEDDING_DIM
            or not np.isfinite(vector).all()
        ):
            invalid.append(key)
            continue

        result[key] = vector.copy()

    if invalid:

        conn.executemany("""
            DELETE FROM prototype_embeddings
            WHERE model_key = ? AND prototype_key = ?
        """, [
            (MODEL_KEY, key)
            for key in invalid
        ])

        conn.commit()

    return result


def save_review_vectors(conn, hashes, vectors):

    records = []

    for key, vector in zip(hashes, vectors):

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        if (
            len(vector) != EMBEDDING_DIM
            or not np.isfinite(vector).all()
        ):
            raise ValueError(
                "Invalid review embedding."
            )

        records.append((
            MODEL_KEY,
            key,
            EMBEDDING_DIM,
            sqlite3.Binary(vector.tobytes())
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO review_embeddings
        (model_key, text_hash, dimension, embedding)
        VALUES (?, ?, ?, ?)
    """, records)

    conn.commit()


def save_prototype_vectors(conn, records):

    conn.executemany("""
        INSERT OR REPLACE INTO prototype_embeddings
        (model_key, prototype_key, category, dimension, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, records)

    conn.commit()


def get_cache_size(conn):

    review_count = conn.execute("""
        SELECT COUNT(*)
        FROM review_embeddings
        WHERE model_key = ?
    """, (MODEL_KEY,)).fetchone()[0]

    prototype_count = conn.execute("""
        SELECT COUNT(*)
        FROM prototype_embeddings
        WHERE model_key = ?
    """, (MODEL_KEY,)).fetchone()[0]

    return {
        "review_vectors": int(review_count),
        "prototype_vectors": int(prototype_count)
    }


# ============================================================
# 10. LOAD LOCAL MINILM
# ============================================================

def load_minilm():

    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download

    except ImportError as exc:

        raise RuntimeError(
            "MiniLM dependencies are missing. "
            "Check the active virtual environment."
        ) from exc

    print("\nLoading local MiniLM...")

    local_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=True
    )

    model = SentenceTransformer(
        str(local_path),
        device=DEVICE
    )

    print("MiniLM loaded successfully.")
    print(f"Device: {DEVICE}")
    print("Mode: Offline / local cache")

    return model


# ============================================================
# 11. PROTOTYPE CACHE
# ============================================================

def build_prototype_specs():

    specs = []

    for category, descriptions in CATEGORY_PROTOTYPES.items():

        for index, description in enumerate(descriptions):

            key = text_hash(
                f"{category}|{index}|{description}"
            )

            specs.append({
                "category": category,
                "index": index,
                "text": description,
                "key": key
            })

    return specs


# ============================================================
# 12. REVIEW + PROTOTYPE VECTOR CACHE
# ============================================================

def get_or_create_embeddings(reviews, conn):

    hashes = (
        reviews["text_hash"].astype(str).tolist()
    )

    normalized_texts = (
        reviews["normalized_text"].astype(str).tolist()
    )

    # Load all existing review vectors.
    review_cache = fetch_review_cache(conn)

    missing_indices = [
        i for i, key in enumerate(hashes)
        if key not in review_cache
    ]

    review_hits = len(hashes) - len(missing_indices)
    review_misses = len(missing_indices)

    # Load prototype vectors.
    prototype_specs = build_prototype_specs()

    prototype_cache = fetch_prototype_cache(conn)

    missing_prototypes = [
        spec for spec in prototype_specs
        if spec["key"] not in prototype_cache
    ]

    prototype_hits = (
        len(prototype_specs) - len(missing_prototypes)
    )

    prototype_misses = len(missing_prototypes)

    print("\n========== MINILM SQLITE CACHE ==========")

    print(f"Database            : {CACHE_DB}")
    print(f"Review cache hits   : {review_hits:,}")
    print(f"Review cache misses : {review_misses:,}")
    print(f"Prototype hits      : {prototype_hits:,}")
    print(f"Prototype misses    : {prototype_misses:,}")

    # Load MiniLM only if something actually needs encoding.
    if review_misses or prototype_misses:

        model = load_minilm()

        # Encode only new or changed review texts.
        if review_misses:

            missing_texts = [
                normalized_texts[i]
                for i in missing_indices
            ]

            missing_hashes = [
                hashes[i]
                for i in missing_indices
            ]

            print(
                f"\nEncoding NEW reviews: "
                f"{len(missing_texts):,}"
            )

            vectors = model.encode(
                missing_texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            ).astype(np.float32)

            if len(vectors) != len(missing_texts):
                raise ValueError(
                    "Review embedding count mismatch."
                )

            save_review_vectors(
                conn,
                missing_hashes,
                vectors
            )

            for key, vector in zip(
                missing_hashes,
                vectors
            ):
                review_cache[key] = vector

        # Encode only new or changed prototypes.
        if prototype_misses:

            prototype_texts = [
                spec["text"]
                for spec in missing_prototypes
            ]

            print(
                f"\nEncoding NEW prototypes: "
                f"{len(prototype_texts)}"
            )

            vectors = model.encode(
                prototype_texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(np.float32)

            if len(vectors) != len(prototype_texts):
                raise ValueError(
                    "Prototype embedding count mismatch."
                )

            records = []

            for spec, vector in zip(
                missing_prototypes,
                vectors
            ):

                records.append((
                    MODEL_KEY,
                    spec["key"],
                    spec["category"],
                    EMBEDDING_DIM,
                    sqlite3.Binary(vector.tobytes())
                ))

                prototype_cache[
                    spec["key"]
                ] = vector

            save_prototype_vectors(
                conn,
                records
            )

        del model
        gc.collect()

    else:

        print(
            "\nAll vectors found in SQLite."
        )

        print(
            "MiniLM loading and encoding skipped."
        )

    # Restore review vectors in exact current review order.
    review_matrix = np.vstack([
        review_cache[key]
        for key in hashes
    ]).astype(np.float32)

    # Restore prototypes in exact category/index order.
    prototype_matrix = np.vstack([
        prototype_cache[spec["key"]]
        for spec in prototype_specs
    ]).astype(np.float32)

    prototype_indexes = {}

    for i, spec in enumerate(prototype_specs):

        prototype_indexes.setdefault(
            spec["category"],
            []
        ).append(i)

    cache_size = get_cache_size(conn)

    cache_stats = {

        "database": str(CACHE_DB),

        "model_key": MODEL_KEY,

        "embedding_dimension": EMBEDDING_DIM,

        "review_cache_hits": review_hits,

        "review_cache_misses": review_misses,

        "prototype_cache_hits": prototype_hits,

        "prototype_cache_misses": prototype_misses,

        "total_cached_review_vectors": (
            cache_size["review_vectors"]
        ),

        "total_cached_prototype_vectors": (
            cache_size["prototype_vectors"]
        )
    }

    print("\n========== CACHE RESULT ==========")

    print(
        f"Stored review vectors : "
        f"{cache_size['review_vectors']:,}"
    )

    print(
        f"Stored prototype vectors : "
        f"{cache_size['prototype_vectors']:,}"
    )

    return (
        review_matrix,
        prototype_matrix,
        prototype_indexes,
        cache_stats
    )


# ============================================================
# 13. SEMANTIC SCORING
# ============================================================

def calculate_semantic_scores(
    reviews,
    conn
):

    if reviews.empty:
        return {}, {}

    (
        review_matrix,
        prototype_matrix,
        prototype_indexes,
        cache_stats
    ) = get_or_create_embeddings(
        reviews,
        conn
    )

    print(
        f"\nCalculating similarity for "
        f"{len(reviews):,} reviews..."
    )

    similarity = (
        review_matrix @ prototype_matrix.T
    )

    results = {}

    for key, row in zip(
        reviews["normalized_text"].tolist(),
        similarity
    ):

        category_scores = {}

        for category, indexes in prototype_indexes.items():

            category_scores[category] = float(
                np.max(row[indexes])
            )

        results[key] = category_scores

    return results, cache_stats


# ============================================================
# 14. SEMANTIC CANDIDATE SELECTION
# ============================================================

def select_semantic_candidates(scores):

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    qualified = [
        (category, score)
        for category, score in ranked
        if (
            category != "packaging"
            and score >= CATEGORY_THRESHOLDS[category]
        )
    ]

    if not qualified:
        return [], "below_threshold"

    top_category, top_score = qualified[0]

    second_score = (
        qualified[1][1]
        if len(qualified) > 1
        else 0.0
    )

    margin = top_score - second_score

    if margin < MIN_TOP_MARGIN:
        return [], "low_top_margin"

    selected = [top_category]

    for category, score in qualified[1:]:

        if (
            top_score - score <= SECONDARY_MARGIN
            and len(selected) < 3
        ):
            selected.append(category)

    return selected, "accepted"

# ============================================================
# 15. SIMPLE STANDALONE SENTIMENT
# ============================================================

SIMPLE_SENTIMENT = {

    "ruim": -0.6,
    "pessimo": -0.9,
    "pessima": -0.9,
    "horrivel": -0.9,
    "terrivel": -0.9,
    "otimo": 0.8,
    "otima": 0.8,
    "excelente": 0.9,
    "bom": 0.5,
    "boa": 0.5,
    "gostei": 0.7,
    "nao gostei": -0.7,
    "nao recomendo": -0.8,
    "recomendo": 0.7,
    "produto bom": 0.5,
    "produto ruim": -0.6,
    "produto excelente": 0.9,
    "produto pessimo": -0.9,
    "gostei do produto": 0.7,
    "horrivel pessimo": -0.9
}


def classify_simple_sentiment(review_id, text):

    normalized = normalize_text(text)

    if normalized not in SIMPLE_SENTIMENT:
        return None

    return {

        "review_id": review_id,

        "original_text": text,

        "routing_decision": "python_simple",

        "rule_status": "standalone_sentiment",

        "rule_categories": "[]",

        "semantic_categories": "[]",

        "candidate_categories": "[]",

        "evidence_quotes": "[]",

        "semantic_scores": "{}",

        "semantic_top_category": "",

        "semantic_top_score": 0.0,

        "semantic_margin": 0.0,

        "delivery_state": "",

        "reason": "Exact standalone sentiment match.",

        "sentiment_score": SIMPLE_SENTIMENT[normalized]
    }


# ============================================================
# 16. PYTHON RULE DETECTION
# ============================================================

def split_sentences(text):

    parts = re.split(
        r"(?<=[.!?;])\s+|\n+",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def detect_issue_candidates(original_text):

    candidates = {}

    for sentence in split_sentences(original_text):

        normalized = normalize_text(sentence)

        for category, patterns in ISSUE_RULES.items():

            matches = [
                pattern
                for pattern in patterns
                if re.search(pattern, normalized)
            ]

            if not matches:
                continue

            if category not in candidates:

                candidates[category] = {
                    "evidence_quotes": [],
                    "matched_patterns": []
                }

            if sentence not in candidates[category][
                "evidence_quotes"
            ]:

                candidates[category][
                    "evidence_quotes"
                ].append(sentence)

            candidates[category][
                "matched_patterns"
            ].extend(matches)

    return candidates


# ============================================================
# 17. CONTEXT DETECTION
# ============================================================

UNCERTAINTY_PATTERNS = [

    r"\btalvez\b",
    r"\bnao sei se\b",
    r"\bacho que\b",
    r"\bnao tenho certeza\b",
    r"\bnao consigo confirmar\b",
    r"\bparece que\b"
]


def has_uncertainty(text):

    normalized = normalize_text(text)

    return any(
        re.search(pattern, normalized)
        for pattern in UNCERTAINTY_PATTERNS
    )


def has_contradictory_product_statement(text):

    normalized = normalize_text(text)

    positive_patterns = [

        r"\bproduto corresponde ao anuncio\b",
        r"\bproduto e igual ao anunciado\b",
        r"\bveio exatamente como anunciado\b",
        r"\bproduto conforme a foto\b",
        r"\bproduto igual a foto\b",
        r"\bproduto correto\b",
        r"\brecebi exatamente o que comprei\b"
    ]

    return any(
        re.search(pattern, normalized)
        for pattern in positive_patterns
    )


# ============================================================
# 18. PYTHON + MINILM FUSION
# ============================================================

def classify_review(row, semantic_scores):

    review_id = str(row["review_id"])
    original_text = str(row["original_text"])
    normalized = normalize_text(original_text)

    # A. Simple standalone sentiment.
    simple = classify_simple_sentiment(
        review_id,
        original_text
    )

    if simple is not None:
        return simple

    # B. Python explicit rules.
    rule_candidates = detect_issue_candidates(
        original_text
    )

    rule_categories = list(
        rule_candidates.keys()
    )

    evidence_quotes = []

    for data in rule_candidates.values():

        for quote in data["evidence_quotes"]:

            if quote not in evidence_quotes:
                evidence_quotes.append(quote)

    # C. MiniLM semantic candidates.
    scores = semantic_scores.get(
        str(row["normalized_text"]),
        {}
    )

    semantic_categories, semantic_status = (
        select_semantic_candidates(scores)
    )

    # D. Business-specific safeguards.

    delivery_state = detect_delivery_state(
        original_text
    )

    # Do not classify an unexplained cancellation
    # as delivery delay.
    if cancellation_without_reason(original_text):

        semantic_categories = [
            c for c in semantic_categories
            if c != "delivery_late"
        ]

    # Explicit positive delivery experience overrides
    # semantic-only delivery candidates.
    if positive_delivery_without_delay(original_text):

        semantic_categories = [
            c for c in semantic_categories
            if c != "delivery_late"
        ]

    # If the product was never received, do not infer
    # wrong_product unless explicit mismatch evidence exists.
    if (
        delivery_state in [
            "not_received",
            "delayed_and_not_received"
        ]
        and "wrong_product" not in rule_categories
    ):

        semantic_categories = [
            c for c in semantic_categories
            if c != "wrong_product"
        ]

    # Packaging is only allowed through explicit
    # Python physical-damage evidence.
    semantic_categories = [
        c for c in semantic_categories
        if c != "packaging"
    ]

    # E. Merge candidates.
    candidates = list(
        dict.fromkeys(
            rule_categories + semantic_categories
        )
    )

    # F. Contradiction checks.
    contradiction = (
        "wrong_product" in candidates
        and has_contradictory_product_statement(
            original_text
        )
    )

    uncertainty = has_uncertainty(normalized)

    # G. Semantic ranking.
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_category = (
        ranked[0][0] if ranked else ""
    )

    top_score = (
        ranked[0][1] if ranked else 0.0
    )

    second_score = (
        ranked[1][1] if len(ranked) > 1 else 0.0
    )

    margin = top_score - second_score

    serialized_scores = json.dumps(
        {
            k: round(float(v), 4)
            for k, v in ranked
        },
        ensure_ascii=False
    )

    common = {

        "review_id": review_id,

        "original_text": original_text,

        "rule_categories": json.dumps(
            rule_categories,
            ensure_ascii=False
        ),

        "semantic_categories": json.dumps(
            semantic_categories,
            ensure_ascii=False
        ),

        "candidate_categories": json.dumps(
            candidates,
            ensure_ascii=False
        ),

        "evidence_quotes": json.dumps(
            evidence_quotes,
            ensure_ascii=False
        ),

        "semantic_scores": serialized_scores,

        "semantic_top_category": top_category,

        "semantic_top_score": round(
            float(top_score), 4
        ),

        "semantic_margin": round(
            float(margin), 4
        ),

        "delivery_state": delivery_state,

        "sentiment_score": None
    }

    # H. No reliable candidate.
    if not candidates:

        return {
            **common,
            "routing_decision": "llm_full",
            "rule_status": "no_candidate",
            "reason": semantic_status
        }

    # I. Contradictory explicit statements.
    if contradiction:

        return {
            **common,
            "routing_decision": "llm_full",
            "rule_status": "explicit_contradiction",
            "reason": "Candidate conflicts with explicit text."
        }

    # J. Uncertain factual statements.
    if uncertainty:

        return {
            **common,
            "routing_decision": "llm_full",
            "rule_status": "uncertain_statement",
            "reason": "Uncertain factual statement."
        }

    # K. Explicit Python evidence takes priority.
    # A disagreement with MiniLM does not automatically
    # force Full. LLM will verify all candidate categories.
    if rule_categories and semantic_categories:

        status = "fused_candidate"
        reason = "Python and MiniLM support candidates."

    elif rule_categories:

        status = "python_candidate"
        reason = "Explicit Python rule candidate."

    else:

        status = "minilm_candidate"
        reason = "MiniLM semantic candidate."

    return {
        **common,
        "routing_decision": "llm_enrich",
        "rule_status": status,
        "reason": reason
    }


# ============================================================
# 19. RUN FUSION ENGINE
# ============================================================

def run_rule_engine(unique, conn):

    print(
        "\n========== PYTHON + MINILM FUSION V2.4 =========="
    )

    simple_results = {}
    remaining_rows = []

    for index, row in unique.iterrows():

        result = classify_simple_sentiment(
            str(row["review_id"]),
            str(row["original_text"])
        )

        if result is not None:
            simple_results[index] = result

        else:
            remaining_rows.append(row)

    remaining = pd.DataFrame(
        remaining_rows,
        columns=unique.columns
    )

    print(f"Unique reviews     : {len(unique):,}")
    print(f"Python simple      : {len(simple_results):,}")
    print(f"Reviews for MiniLM : {len(remaining):,}")

    semantic_scores = {}
    cache_stats = {}

    if not remaining.empty:

        semantic_scores, cache_stats = (
            calculate_semantic_scores(
                remaining,
                conn
            )
        )

    predictions = []

    for index, row in unique.iterrows():

        if index in simple_results:
            result = simple_results[index]

        else:
            result = classify_review(
                row,
                semantic_scores
            )

        predictions.append(result)

    result_df = pd.DataFrame(predictions)

    # --------------------------------------------------------
    # Routing summary
    # --------------------------------------------------------

    total = len(result_df)

    route_counts = (
        result_df["routing_decision"].value_counts()
    )

    print("\nRouting summary:")

    for route in ROUTES:

        count = int(
            route_counts.get(route, 0)
        )

        ratio = (
            count / total * 100
            if total else 0
        )

        print(
            f"  {route:<15}: "
            f"{count:>7,} ({ratio:>5.1f}%)"
        )

    # --------------------------------------------------------
    # Candidate status
    # --------------------------------------------------------

    print("\nCandidate status:")

    status_counts = (
        result_df["rule_status"].value_counts()
    )

    for status, count in status_counts.items():

        print(
            f"  {status:<34}: {count:>7,}"
        )

    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------

    category_counts = Counter()

    for value in result_df[
        "candidate_categories"
    ].dropna():

        try:
            category_counts.update(
                json.loads(value)
            )
        except (ValueError, TypeError):
            continue

    print("\nMerged candidate categories:")

    for category in CATEGORIES:

        print(
            f"  {category:<24}: "
            f"{category_counts.get(category, 0):>7,}"
        )

    # --------------------------------------------------------
    # Delivery state
    # --------------------------------------------------------

    print("\nInternal delivery state:")

    delivery_counts = (
        result_df["delivery_state"].value_counts()
    )

    for state, count in delivery_counts.items():

        print(
            f"  {state or '(not identified)':<30}: "
            f"{count:>7,}"
        )

    # --------------------------------------------------------
    # Routing examples
    # --------------------------------------------------------

    print("\nRouting examples:")

    for route in ROUTES:

        subset = result_df.loc[
            result_df["routing_decision"] == route
        ]

        if subset.empty:
            continue

        samples = subset.sample(
            n=min(3, len(subset)),
            random_state=42
        )

        print(f"\n  [{route}]")

        for _, sample in samples.iterrows():

            text = str(
                sample["original_text"]
            ).replace("\n", " ")

            if len(text) > 140:
                text = text[:137] + "..."

            print(f"    - {text}")

            print(
                f"      candidates: "
                f"{sample['candidate_categories']}"
            )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    csv_path = (
        RULE_OUTPUT_DIR
        / "rule_predictions_v2_4.csv"
    )

    json_path = (
        RULE_OUTPUT_DIR
        / "rule_predictions_v2_4.json"
    )

    diagnostics_path = (
        RULE_OUTPUT_DIR
        / "rule_diagnostics_v2_4.json"
    )

    save_csv(result_df, csv_path)

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f,
            ensure_ascii=False,
            indent=2
        )

    report = {

        "version": VERSION,

        "method": "Python Rules + MiniLM Fusion",

        "model": MODEL_NAME,

        "model_key": MODEL_KEY,

        "device": DEVICE,

        "category_thresholds": CATEGORY_THRESHOLDS,

        "min_top_margin": MIN_TOP_MARGIN,

        "secondary_margin": SECONDARY_MARGIN,

        "threshold_note": (
            "Initial cosine similarity thresholds, "
            "not calibrated probabilities."
        ),

        "packaging_definition": (
            "Only explicit physical damage to the "
            "packaging itself. MiniLM cannot independently "
            "create packaging candidates."
        ),

        "delivery_definition": (
            "Cancellation without explicit delivery evidence "
            "is not automatically classified as delivery delay."
        ),

        "embedding_cache": cache_stats,

        "total_unique_reviews": total,

        "routing_distribution": {
            route: int(
                route_counts.get(route, 0)
            )
            for route in ROUTES
        },

        "routing_percentages": {
            route: round(
                route_counts.get(route, 0) / total * 100,
                2
            ) if total else 0
            for route in ROUTES
        },

        "candidate_status_distribution": {
            str(k): int(v)
            for k, v in status_counts.items()
        },

        "candidate_category_counts": {
            category: int(
                category_counts.get(category, 0)
            )
            for category in CATEGORIES
        },

        "delivery_state_distribution": {
            str(k): int(v)
            for k, v in delivery_counts.items()
        }
    }

    with open(
        diagnostics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFusion completed.")

    print(f"CSV         : {csv_path}")
    print(f"JSON        : {json_path}")
    print(f"Diagnostics : {diagnostics_path}")

    return result_df, report


# ============================================================
# 20. DEEPSEEK API + JSON EXTRACTION V3.1
# Python + MiniLM + DeepSeek
# Resumable SQLite Cache / Concurrent API / Strict JSON Schema
# ============================================================

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path

import requests
import sqlite3
import json
import re


# ------------------------------------------------------------
# 20.1 API CONFIGURATION
# ------------------------------------------------------------

load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("YEELO_API_KEY")
BASE_URL = os.getenv("BASE_URL")
API_MODEL = os.getenv("MODEL_NAME")

API_URL = (
    BASE_URL.rstrip("/") + "/chat/completions"
    if BASE_URL else ""
)

LLM_CACHE_DB = CACHE_DIR / "llm_results.sqlite3"
PROMPT_TEMPLATE_PATH = BASE_DIR / "prompt_template.txt"

LLM_PROMPT_VERSION = "v3.2-cn"
LLM_SCHEMA_VERSION = "v1"

# First run: test 20 records.
# After successful verification, change to None.
LLM_LIMIT = 20  # 小规模试运行：默认最多处理 20 条唯一文本。

LLM_CONCURRENCY = 3
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 120

# SQLite write lock.
LLM_DB_LOCK = threading.Lock()

ALLOWED_CATEGORIES = {
    "delivery_late",
    "wrong_product",
    "quality_defect",
    "missing_item",
    "seller_communication",
    "packaging",
    "other"
}

ALLOWED_SEVERITIES = {"low", "medium", "high"}


# ------------------------------------------------------------
# 20.2 API CONFIGURATION VALIDATION
# ------------------------------------------------------------

def validate_api_config():

    if not all([API_KEY, BASE_URL, API_MODEL]):

        raise ValueError(
            "Missing Yeellow API configuration. "
            "Check YEELO_API_KEY, BASE_URL and MODEL_NAME."
        )

    print("\n========== DEEPSEEK API ==========")
    print(f"Gateway : {BASE_URL}")
    print(f"Model   : {API_MODEL}")
    print("API Key : Loaded")


# ------------------------------------------------------------
# 20.3 SQLITE LLM CACHE
# ------------------------------------------------------------

def initialize_llm_cache(conn):

    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_results (
            cache_key TEXT PRIMARY KEY,
            prompt_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            result_json TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


def make_llm_cache_key(text):

    raw = "|".join([
        LLM_PROMPT_VERSION,
        LLM_SCHEMA_VERSION,
        API_MODEL or "",
        normalize_text(text)
    ])

    return text_hash(raw)


def get_cached_llm_result(conn, cache_key):

    with LLM_DB_LOCK:

        row = conn.execute("""
            SELECT result_json
            FROM llm_results
            WHERE cache_key = ?
        """, (cache_key,)).fetchone()

    if row is None:
        return None

    try:
        return json.loads(row[0])

    except (ValueError, TypeError):
        return None


def save_llm_result(
    conn,
    cache_key,
    result,
    input_tokens=0,
    output_tokens=0
):

    serialized = json.dumps(
        result,
        ensure_ascii=False
    )

    with LLM_DB_LOCK:

        conn.execute("""
            INSERT OR REPLACE INTO llm_results
            (
                cache_key,
                prompt_version,
                model_name,
                result_json,
                input_tokens,
                output_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            cache_key,
            LLM_PROMPT_VERSION,
            API_MODEL,
            serialized,
            input_tokens,
            output_tokens
        ))

        conn.commit()


# ------------------------------------------------------------
# 20.4 STRICT JSON SCHEMA
# ------------------------------------------------------------

def build_llm_prompt(
    review_id,
    text,
    candidates,
    delivery_state
):
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {PROMPT_TEMPLATE_PATH}"
        )

    system_prompt = PROMPT_TEMPLATE_PATH.read_text(
        encoding="utf-8"
    ).strip()

    user_prompt = {
        "review_id": str(review_id),
        "original_review": text,
        "python_minilm_candidates": candidates,
        "internal_delivery_state": delivery_state,
        "instruction": (
            "请独立分析葡萄牙语原文。候选类别仅作为辅助线索，"
            "不能当作已确认事实。必须遵守 System Prompt 中的分类边界，"
            "不得编造原文没有的信息。只返回规定的 JSON 对象。"
        )
    }

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": json.dumps(
                user_prompt,
                ensure_ascii=False
            )
        }
    ]


# ------------------------------------------------------------
# 20.5 JSON PARSING
# ------------------------------------------------------------

def parse_llm_json(content):

    content = content.strip()

    if content.startswith("```"):

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

    result = json.loads(content)

    if not isinstance(result, dict):

        raise ValueError(
            "LLM output must be a JSON object."
        )

    return result


# ------------------------------------------------------------
# 20.6 JSON VALIDATION
# ------------------------------------------------------------

def validate_llm_result(result, original_text, review_id):

    required = {
        "review_id",
        "issues",
        "sentiment_score",
        "actionable_for_seller",
        "confidence"
    }

    if set(result.keys()) != required:

        raise ValueError(
            "Invalid top-level JSON schema."
        )

    if str(result["review_id"]) != str(review_id):

        raise ValueError(
            "Returned review_id does not match input."
        )

    if not isinstance(result["issues"], list):

        raise ValueError(
            "issues must be a list."
        )

    score = result["sentiment_score"]
    confidence = result["confidence"]

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not -1 <= score <= 1
    ):

        raise ValueError(
            "Invalid sentiment_score."
        )

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):

        raise ValueError(
            "Invalid confidence."
        )

    if not isinstance(
        result["actionable_for_seller"],
        bool
    ):

        raise ValueError(
            "actionable_for_seller must be boolean."
        )

    for issue in result["issues"]:

        if not isinstance(issue, dict):

            raise ValueError(
                "Invalid issue object."
            )

        expected = {
            "category",
            "severity",
            "evidence_quote",
            "evidence_quote_en"
        }

        if set(issue.keys()) != expected:

            raise ValueError(
                "Invalid issue schema."
            )

        if issue["category"] not in ALLOWED_CATEGORIES:

            raise ValueError(
                "Invalid category."
            )

        if issue["severity"] not in ALLOWED_SEVERITIES:

            raise ValueError(
                "Invalid severity."
            )

        quote = issue["evidence_quote"]

        if not isinstance(quote, str) or not quote.strip():

            raise ValueError(
                "Missing Portuguese evidence."
            )

        if quote not in original_text:

            raise ValueError(
                "Evidence quote is not an exact substring "
                "of original review."
            )

        translation = issue["evidence_quote_en"]

        if not isinstance(translation, str) or not translation.strip():

            raise ValueError(
                "Missing English translation."
            )

    return result


# ------------------------------------------------------------
# 20.7 DEEPSEEK REQUEST
# ------------------------------------------------------------

def call_deepseek(
    review_id,
    text,
    candidates,
    delivery_state
):

    messages = build_llm_prompt(
        review_id,
        text,
        candidates,
        delivery_state
    )

    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):

        try:

            response = requests.post(
                API_URL,

                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },

                json={
                    "model": API_MODEL,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 1500,
                    "response_format": {
                        "type": "json_object"
                    }
                },

                timeout=LLM_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            content = data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

            result = parse_llm_json(content)

            usage = data.get("usage", {})

            return result, {
                "input_tokens": usage.get(
                    "prompt_tokens", 0
                ),
                "output_tokens": usage.get(
                    "completion_tokens", 0
                )
            }

        except Exception as exc:

            last_error = exc

            print(
                f"\nAPI attempt {attempt}/"
                f"{LLM_MAX_RETRIES} failed: {exc}"
            )

            if attempt < LLM_MAX_RETRIES:

                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"DeepSeek request failed: {last_error}"
    )


# ------------------------------------------------------------
# 20.8 PROCESS ONE REVIEW
# ------------------------------------------------------------

def process_one_review(row, conn):

    review_id = str(row["review_id"])
    original_text = str(row["original_text"])
    normalized = str(row["normalized_text"])

    cache_key = make_llm_cache_key(normalized)

    cached = get_cached_llm_result(
        conn,
        cache_key
    )

    if cached is not None:

        cached_result = {
            "review_id": review_id,
            **cached
        }

        validate_llm_result(
            cached_result,
            original_text,
            review_id
        )

        return cached_result, "CACHE", 0, 0

    candidates = json.loads(
        row["candidate_categories"]
    )

    delivery_state = str(
        row["delivery_state"]
    )

    messages = build_llm_prompt(
        review_id,
        original_text,
        candidates,
        delivery_state
    )

    last_validation_error = None

    for validation_attempt in range(2):

        if validation_attempt == 0:

            result, usage = call_deepseek(
                review_id,
                original_text,
                candidates,
                delivery_state
            )

        else:

            messages.append({
                "role": "assistant",
                "content": json.dumps(
                    result,
                    ensure_ascii=False
                )
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous JSON failed validation: "
                    f"{last_validation_error}. "
                    "Correct the JSON. "
                    "Return exactly the required schema. "
                    "Every Portuguese evidence quote must "
                    "be copied exactly from the original review."
                )
            })

            corrected = False

            for api_attempt in range(
                1,
                LLM_MAX_RETRIES + 1
            ):

                try:

                    response = requests.post(
                        API_URL,

                        headers={
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json"
                        },

                        json={
                            "model": API_MODEL,
                            "messages": messages,
                            "temperature": 0,
                            "max_tokens": 1500,
                            "response_format": {
                                "type": "json_object"
                            }
                        },

                        timeout=LLM_TIMEOUT
                    )

                    response.raise_for_status()

                    data = response.json()

                    result = parse_llm_json(
                        data["choices"][0]["message"]["content"]
                    )

                    usage = {
                        "input_tokens": data.get(
                            "usage", {}
                        ).get("prompt_tokens", 0),

                        "output_tokens": data.get(
                            "usage", {}
                        ).get("completion_tokens", 0)
                    }

                    corrected = True
                    break

                except Exception as exc:

                    if api_attempt == LLM_MAX_RETRIES:
                        raise

                    time.sleep(2 ** api_attempt)

            if not corrected:

                raise RuntimeError(
                    "JSON correction failed."
                )

        try:

            validate_llm_result(
                result,
                original_text,
                review_id
            )

            # Cache payload excludes review_id.
            # Identical review texts can reuse the same result.
            cache_payload = {
                "issues": result["issues"],
                "sentiment_score": result["sentiment_score"],
                "actionable_for_seller": result[
                    "actionable_for_seller"
                ],
                "confidence": result["confidence"]
            }

            save_llm_result(
                conn,
                cache_key,
                cache_payload,
                usage["input_tokens"],
                usage["output_tokens"]
            )

            return (
                result,
                "API",
                usage["input_tokens"],
                usage["output_tokens"]
            )

        except ValueError as exc:

            last_validation_error = str(exc)

            if validation_attempt == 1:
                raise

    raise RuntimeError(
        f"Could not obtain valid output: {review_id}"
    )


# ------------------------------------------------------------
# 20.9 RUN LLM EXTRACTION
# ------------------------------------------------------------

def run_llm_extraction(
    valid,
    empty,
    fusion_df,
    conn
):

    print(
        "\n========== DEEPSEEK EXTRACTION V3.2 =========="
    )

    with sqlite3.connect(
        str(LLM_CACHE_DB),
        timeout=60
    ) as llm_init_conn:
        initialize_llm_cache(llm_init_conn)

    validate_api_config()

    fusion_lookup = fusion_df.set_index(
        "review_id"
    )

    unique_valid = valid.drop_duplicates(
        subset=["normalized_text"],
        keep="first"
    ).copy()

    records = []

    for _, row in unique_valid.iterrows():

        review_id = str(row["review_id"])

        if review_id not in fusion_lookup.index:

            raise ValueError(
                f"Missing fusion result: {review_id}"
            )

        record = row.to_dict()

        fusion_record = fusion_lookup.loc[
            review_id
        ].to_dict()

        record.update(fusion_record)

        records.append(record)

    total_unique = len(records)

    if LLM_LIMIT is not None:

        records = records[:LLM_LIMIT]

    print(
        f"Unique non-empty texts : {total_unique:,}"
    )

    print(
        f"Texts selected         : {len(records):,}"
    )

    print(
        f"Concurrency            : {LLM_CONCURRENCY}"
    )

    results_by_text = {}

    cache_hits = 0
    api_calls = 0

    input_tokens = 0
    output_tokens = 0

    failures = []

    def worker(record):

        local_conn = sqlite3.connect(
            str(LLM_CACHE_DB),
            timeout=60
        )

        try:

            local_conn.execute(
                "PRAGMA busy_timeout=60000"
            )

            return process_one_review(
                record,
                local_conn
            )

        finally:

            local_conn.close()

    with ThreadPoolExecutor(
        max_workers=LLM_CONCURRENCY
    ) as executor:

        futures = {
            executor.submit(worker, record): record
            for record in records
        }

        total = len(futures)

        for completed, future in enumerate(
            as_completed(futures),
            start=1
        ):

            record = futures[future]

            try:

                result, source, in_tokens, out_tokens = (
                    future.result()
                )

                normalized = str(
                    record["normalized_text"]
                )

                results_by_text[normalized] = result

                if source == "CACHE":
                    cache_hits += 1

                else:
                    api_calls += 1

                input_tokens += in_tokens
                output_tokens += out_tokens

            except Exception as exc:

                failures.append({
                    "review_id": str(
                        record["review_id"]
                    ),

                    "original_text": str(
                        record["original_text"]
                    ),

                    "error": str(exc)
                })

            if completed % 5 == 0 or completed == total:

                print(
                    f"\rProgress: {completed:,}/{total:,} | "
                    f"API: {api_calls:,} | "
                    f"Cache: {cache_hits:,} | "
                    f"Failed: {len(failures):,}",
                    end="",
                    flush=True
                )

    print("\n")

    # --------------------------------------------------------
    # 将成功处理的唯一文本结果映射回原始 review_id。
    # 空评论、失败评论和未处理评论不伪装成模型成功结果。
    # --------------------------------------------------------

    final_results = []
    processed_ids = set()
    duplicate_review_ids_skipped = 0

    for _, row in valid.iterrows():

        normalized = str(row["normalized_text"])

        if normalized not in results_by_text:
            continue

        review_id = str(row["review_id"])

        # 同一个 review_id 只保留一条，避免最终 JSON 重复。
        if review_id in processed_ids:
            duplicate_review_ids_skipped += 1
            continue

        result = dict(results_by_text[normalized])
        result["review_id"] = review_id

        result = {
            "review_id": result["review_id"],
            "issues": result["issues"],
            "sentiment_score": result["sentiment_score"],
            "actionable_for_seller": result["actionable_for_seller"],
            "confidence": result["confidence"]
        }

        final_results.append(result)
        processed_ids.add(review_id)

    # --------------------------------------------------------
    # Save final output.
    # --------------------------------------------------------

    output_path = (
        RULE_OUTPUT_DIR /
        "extracted_issues.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save failures separately.
    # --------------------------------------------------------

    failure_path = (
        RULE_OUTPUT_DIR /
        "llm_failures.json"
    )

    with open(
        failure_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            failures,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    report = {

        "version": "3.2",

        "model": API_MODEL,

        "prompt_version": LLM_PROMPT_VERSION,

        "schema_version": LLM_SCHEMA_VERSION,

        "total_unique_texts": total_unique,

        "selected_unique_texts": len(records),

        "api_calls": api_calls,

        "cache_hits": cache_hits,

        "failures": len(failures),

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "mapped_nonempty_reviews": len(final_results),
        "duplicate_review_ids_skipped": duplicate_review_ids_skipped,
        "successful_unique_texts": len(results_by_text),

        "empty_reviews": len(empty),

        "final_records": len(final_results),

        "unprocessed_unique_texts": (
            total_unique - len(results_by_text)
        ),

        "final_output": str(output_path),

        "failure_output": str(failure_path)
    }

    report_path = (
        RULE_OUTPUT_DIR /
        "llm_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\n========== DEEPSEEK SUMMARY =========="
    )

    print(f"API calls      : {api_calls:,}")
    print(f"Cache hits     : {cache_hits:,}")
    print(f"Failures       : {len(failures):,}")
    print(f"Input tokens   : {input_tokens:,}")
    print(f"Output tokens  : {output_tokens:,}")
    print(f"Final records  : {len(final_results):,}")

    print(f"JSON           : {output_path}")
    print(f"Failures       : {failure_path}")
    print(f"Report         : {report_path}")

    return final_results, report


# ============================================================
# 21. MAIN PIPELINE V3.1
# ============================================================

def main():

    print("\n" + "=" * 60)

    print(
        " Q4 | REVIEW EXTRACTION PIPELINE V3.2"
    )

    print(
        " Python + MiniLM + DeepSeek"
    )

    print("=" * 60)

    print("\n[1/6] DATA LOADING")

    source = load_data()

    print("\n[2/6] NEGATIVE REVIEW FILTERING")

    negative = filter_reviews(source)

    print("\n[3/6] DATA PREPROCESSING")

    all_reviews, valid, empty, unique = (
        prepare_reviews(negative)
    )

    print("\n[4/6] EXPORT & VALIDATION")

    export_preprocessing(
        all_reviews,
        valid,
        empty,
        unique
    )

    print("\n[5/6] PYTHON + MINILM FUSION")

    conn = open_cache()

    try:

        fusion_df, fusion_report = (
            run_rule_engine(
                unique,
                conn
            )
        )

        print("\n[6/6] DEEPSEEK EXTRACTION")

        final_results, llm_report = (
            run_llm_extraction(
                valid,
                empty,
                fusion_df,
                conn
            )
        )

    finally:

        conn.close()

    # --------------------------------------------------------
    # Final pipeline report.
    # --------------------------------------------------------

    pipeline_report = {

        "pipeline_version": "3.2",

        "source_records": len(source),

        "negative_reviews": len(negative),

        "valid_comments": len(valid),

        "empty_comments": len(empty),

        "unique_texts": len(unique),

        "duplicate_text_rows": (
            len(valid) - len(unique)
        ),

        "preprocessing_integrity": "PASSED",

        "fusion": fusion_report,

        "llm_extraction": llm_report,

        "final_json_records": len(final_results),

        "embedding_database": str(CACHE_DB),

        "llm_database": str(LLM_CACHE_DB)
    }

    report_path = (
        RULE_OUTPUT_DIR /
        "pipeline_report_v3_2.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pipeline_report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)

    print(
        " PIPELINE V3.1 FINISHED"
    )

    print("=" * 60)

    print(f"Source records : {len(source):,}")
    print(f"Negative       : {len(negative):,}")
    print(f"Unique texts   : {len(unique):,}")
    print(f"Final records  : {len(final_results):,}")

    print(f"Vector DB      : {CACHE_DB}")
    print(f"LLM DB         : {LLM_CACHE_DB}")

    print(f"Report         : {report_path}")


if __name__ == "__main__":
    main()
