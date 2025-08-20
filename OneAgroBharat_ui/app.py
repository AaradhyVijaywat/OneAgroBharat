# app.py
import os
import gzip
import io
import traceback
from typing import Dict, List, Optional

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
)
import joblib
import pandas as pd

# --------- Configuration ----------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(APP_ROOT, "model.pkl.gz"),
    os.path.join(APP_ROOT, "model.pkl"),
    os.path.join(APP_ROOT, "model.joblib"),
    os.path.join(APP_ROOT, "model.sav"),
]
PREPROCESSOR_CANDIDATES = [
    os.path.join(APP_ROOT, "preprocessor.pkl.gz"),
    os.path.join(APP_ROOT, "preprocessor.pkl"),
    os.path.join(APP_ROOT, "preprocessor.joblib"),
]
DATASET_PATH = os.path.join(APP_ROOT, "yield_df.csv")

# --------- Flask app ----------
app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-key"

# --------- Utilities: load model/preprocessor/dataset ----------
def load_model():
    """Try to load model from common places (gz/pkl/joblib)."""
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                if p.endswith(".pkl.gz"):
                    with gzip.open(p, "rb") as f:
                        return joblib.load(f)
                else:
                    return joblib.load(p)
            except Exception as e:
                app.logger.error(f"Error loading model from {p}: {e}")
                continue
    return None


def load_preprocessor():
    """Load preprocessor if present (gz/pkl/joblib)."""
    for p in PREPROCESSOR_CANDIDATES:
        if os.path.exists(p):
            try:
                if p.endswith(".pkl.gz"):
                    with gzip.open(p, "rb") as f:
                        return joblib.load(f)
                else:
                    return joblib.load(p)
            except Exception as e:
                app.logger.error(f"Error loading preprocessor from {p}: {e}")
                continue
    return None


def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        # drop stray unnamed index columns
        unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df
    return None


def detect_target_col(df: pd.DataFrame) -> Optional[str]:
    if df is None:
        return None
    candidates = ["Yield", "yield", "Production", "production", "target", "Target", "label"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_encoders(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build simple label encoders (dict mapping) for object columns.
    Unseen categories during prediction map to -1.
    """
    encoders: Dict[str, Dict[str, int]] = {}
    if df is None:
        return encoders
    for col in feature_cols:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            cats = pd.Series(df[col].astype(str).unique()).sort_values().tolist()
            encoders[col] = {cat: i for i, cat in enumerate(cats)}
    return encoders


def encode_dataframe(df_in: pd.DataFrame, encoders: Dict[str, Dict[str, int]]):
    out = df_in.copy()
    for col, mapping in encoders.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)
    return out


# --------- Load artifacts on start ----------
MODEL = load_model()
PREPROCESSOR = load_preprocessor()
DF = load_dataset()
TARGET_COL = detect_target_col(DF)
FEATURE_COLS = [c for c in DF.columns if c != TARGET_COL] if DF is not None and TARGET_COL else (list(DF.columns) if DF is not None else [])
ENCODERS = build_encoders(DF, FEATURE_COLS) if DF is not None else {}

# Pre-built fallback country/crop list (used if the dataset does not contain a large list)
FALLBACK_COUNTRIES = [
    "India","United States","China","Brazil","Australia","Canada","United Kingdom",
    "Germany","France","Mexico","Pakistan","Bangladesh","Nigeria","Ethiopia","Argentina",
    "Indonesia","Vietnam","Philippines","Thailand","Egypt","Turkey","Spain","Italy"
]

# --------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    """
    Render main page. We pass the feature schema (name/type/options) so the template
    can render inputs dynamically.
    """
    fields = []
    for col in FEATURE_COLS:
        field = {"name": col}
        if col in ENCODERS:
            # categorical
            field["type"] = "categorical"
            field["options"] = sorted(list(ENCODERS[col].keys()))
            # if many unique values and user wants country list, we still show options
        else:
            # numeric fallback
            field["type"] = "numeric"
            # determine default value (median)
            try:
                series = pd.to_numeric(DF[col], errors="coerce")
                default = float(series.median()) if series.notna().any() else 0.0
            except Exception:
                default = 0.0
            field["default"] = default
        fields.append(field)

    # If no dataset fields found, offer some fallback fields (Area etc.)
    if not fields:
        fields = [
            {"name": "Area", "type": "numeric", "default": 1.0},
            {"name": "Country", "type": "categorical", "options": FALLBACK_COUNTRIES},
            {"name": "Rainfall", "type": "numeric", "default": 100.0},
        ]

    return render_template(
        "index.html",
        fields=fields,
        model_loaded=MODEL is not None,
        preprocessor_loaded=PREPROCESSOR is not None,
        dataset_loaded=DF is not None,
        target_col=TARGET_COL,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form submission for a single prediction.
    Validate inputs client & server-side, encode as necessary, predict, and return result.
    """
    if MODEL is None:
        flash("Model is missing. Place model.pkl.gz or model.pkl in project root.", "danger")
        return redirect(url_for("index"))

    # collect form inputs
    try:
        # Build input dict from posted values (we only consider fields displayed in UI)
        input_dict = {}
        for key, val in request.form.items():
            # skip CSRF or non-field keys
            if key.startswith("_"):
                continue
            input_dict[key] = val.strip()
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Failed to read form inputs.", "danger")
        return redirect(url_for("index"))

    # server-side validation & conversion
    df_input = pd.DataFrame([input_dict])

    # Convert numeric columns to float where appropriate; detect numeric by attempting conversion
    for col in df_input.columns:
        # If field exists in encoders, leave as string
        if col in ENCODERS:
            continue
        # Try numeric conversion
        try:
            df_input[col] = pd.to_numeric(df_input[col].astype(str).str.replace(",", ""), errors="coerce")
        except Exception:
            df_input[col] = df_input[col]

    # If preprocessor is available, use it; else use encoders built from dataset
    try:
        if PREPROCESSOR is not None:
            # PREPROCESSOR may be a transformer (ColumnTransformer) expecting DataFrame or array.
            # Many preprocessors expect DataFrame columns in original order; ensure same columns
            X_input = df_input.copy()
            # If any missing features, attempt to add them as NaN
            for c in FEATURE_COLS:
                if c not in X_input.columns:
                    X_input[c] = pd.NA
            X_input = X_input[FEATURE_COLS]
            X_proc = PREPROCESSOR.transform(X_input)
            preds = MODEL.predict(X_proc)
        else:
            # encode categorical using simple mapping (unseen -> -1)
            X_input = df_input.copy()
            # ensure columns order
            if FEATURE_COLS:
                for c in FEATURE_COLS:
                    if c not in X_input.columns:
                        X_input[c] = pd.NA
                X_input = X_input[FEATURE_COLS]
            X_enc = encode_dataframe(X_input, ENCODERS)
            preds = MODEL.predict(X_enc)
    except Exception as e:
        app.logger.error("Prediction error:\n" + traceback.format_exc())
        flash("Prediction failed on the server. Check console logs. Error: " + str(e), "danger")
        return redirect(url_for("index"))

    # return result
    result = float(preds[0]) if hasattr(preds, "__len__") else float(preds)
    return render_template("index.html", prediction=result, fields=[], model_loaded=True, preprocessor_loaded=(PREPROCESSOR is not None), dataset_loaded=(DF is not None))



@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Accept uploaded CSV for batch predictions. Must include same feature columns used by the app.
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 400

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df_in = pd.read_csv(f)
        unnamed = [c for c in df_in.columns if c.lower().startswith("unnamed")]
        if unnamed:
            df_in = df_in.drop(columns=unnamed)

        # ensure required columns exist
        if FEATURE_COLS:
            missing = [c for c in FEATURE_COLS if c not in df_in.columns]
            if missing:
                return jsonify({"error": f"Missing columns: {missing}"}), 400

        # preprocess & predict
        if PREPROCESSOR is not None:
            X_proc = PREPROCESSOR.transform(df_in[FEATURE_COLS])
            preds = MODEL.predict(X_proc)
        else:
            X_enc = encode_dataframe(df_in[FEATURE_COLS], ENCODERS)
            preds = MODEL.predict(X_enc)

        df_out = df_in.copy()
        df_out["Prediction"] = preds
        buf = io.StringIO()
        df_out.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv", as_attachment=True, attachment_filename="predictions.csv")
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Run app
if __name__ == "__main__":
    # For local debugging, enable debug=True. For final submission, you can set debug=False
    app.run(host="0.0.0.0", port=5000, debug=True)
