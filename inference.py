import json
import joblib
import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoTokenizer, AutoModel

MODEL_FOLDER = "model_artifacts_3"
device = torch.device("cpu")


# =====================================================
# LOAD MODEL + ARTIFACTS
# =====================================================
def load_model():
    with open(f"{MODEL_FOLDER}/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    MODEL_NAME = cfg["MODEL_NAME"]
    LEVELS = cfg["LEVELS"]
    MAX_LEN = cfg["MAX_LEN"]

    label_encoders = joblib.load(
        f"{MODEL_FOLDER}/label_encoders.pkl"
    )

    tree_maps = joblib.load(
        f"{MODEL_FOLDER}/tree_maps.pkl"
    )

    num_classes = {
        lvl: len(label_encoders[lvl].classes_)
        for lvl in LEVELS
    }

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True
    )

    # =================================================
    # MODEL CLASS
    # =================================================
    class HMC(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                local_files_only=True
            )

            hidden_size = self.encoder.config.hidden_size

            self.classifiers = nn.ModuleDict({
                lvl: nn.Linear(hidden_size, num_classes[lvl])
                for lvl in LEVELS
            })

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls_emb = outputs.last_hidden_state[:, 0, :]

            return {
                lvl: self.classifiers[lvl](cls_emb)
                for lvl in LEVELS
            }

    # =================================================
    # INIT MODEL
    # =================================================
    model = HMC()

    # model kamu quantized, jadi quantize dulu
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    state_dict = torch.load(
        f"{MODEL_FOLDER}/model_HMC.pt_v.3",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    # =================================================
    # CATEGORY MAPPING
    # =================================================
    mapping_df = pd.read_csv("mappings/category_mapping.csv")

    mapping_df = mapping_df[["id", "parent_category_id", "name"]].copy()

    mapping_df["name"] = (
        mapping_df["name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    mapping_df["parent_category_id"] = (
        mapping_df["parent_category_id"]
        .fillna(-1)
        .astype(int)
    )

    mapping_df["id"] = mapping_df["id"].astype(int)

    # key = (parent_id, category_name)
    category_lookup = {}

    for _, row in mapping_df.iterrows():
        key = (row["parent_category_id"], row["name"])
        category_lookup[key] = row["id"]

    # =================================================
    # PRECOMPUTE MASK CACHE
    # =================================================
    mask_cache = {}

    for (parent_lvl, child_lvl), mapping in tree_maps.items():
        for parent_id, children in mapping.items():

            key = (child_lvl, parent_id)

            mask = torch.full(
                (num_classes[child_lvl],),
                -1e9
            )

            children = list(children)

            if len(children) > 0:
                mask[children] = 0

            mask_cache[key] = mask

    return (
        model,
        tokenizer,
        label_encoders,
        LEVELS,
        MAX_LEN,
        mask_cache,
        category_lookup
    )


# =====================================================
# LOAD ONCE
# =====================================================
(
    model,
    tokenizer,
    label_encoders,
    LEVELS,
    MAX_LEN,
    mask_cache,
    category_lookup
) = load_model()


# =====================================================
# SINGLE PREDICTION
# =====================================================
def predict_text(item_name: str, description: str = ""):
    text = f"{item_name} {description}".lower().strip()

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.inference_mode():
        logits = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

    parent_label_id = None
    parent_company_id = -1   # root
    last_valid_company_id = None

    for lvl_idx, lvl in enumerate(LEVELS):
        logit = logits[lvl][0]

        # hierarchical mask
        if lvl_idx > 0 and parent_label_id is not None:
            key = (lvl, parent_label_id)

            if key in mask_cache:
                logit = logit + mask_cache[key]
            else:
                break

        pred = torch.argmax(logit).item()

        pred_category = (
            str(label_encoders[lvl].classes_[pred])
            .lower()
            .strip()
        )

        # stop kalau prediksi nan
        if pred_category == "nan":
            break

        company_id = category_lookup.get(
            (parent_company_id, pred_category),
            None
        )

        # stop kalau mapping tidak ketemu
        if company_id is None:
            break

        # simpan child terakhir valid
        last_valid_company_id = int(company_id)

        # update parent
        parent_label_id = pred
        parent_company_id = company_id

    return last_valid_company_id

# =====================================================
# CSV PREDICTION
# =====================================================
def predict_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    results = []

    for _, row in df.iterrows():
        item_name = row.get("nama item", "")
        description = row.get("deskripsi", "")

        pred = predict_text(item_name, description)

        flat = {}

        for lvl, value in pred.items():
            flat[lvl] = value["category_name"]
            flat[f"{lvl}_id"] = value["company_id"]

        results.append(flat)

    pred_df = pd.DataFrame(results)

    final_df = pd.concat(
        [df.reset_index(drop=True), pred_df],
        axis=1
    )

    final_df.to_csv(output_csv, index=False)

    return output_csv