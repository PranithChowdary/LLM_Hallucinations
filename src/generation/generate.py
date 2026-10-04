from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_FILE = PROJECT_ROOT / "data" / "calibration.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"

SAVE_EVERY = 100


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Data type: {DTYPE}")

    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load calibration data
    # --------------------------------------------------------

    print("[INFO] Loading calibration data...")

    df = pd.read_parquet(CALIBRATION_FILE)

    print(f"[INFO] Total calibration prompts: {len(df)}")

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        existing_df = pd.read_parquet(OUTPUT_FILE)

        completed_uids = set(existing_df["uid"])

        print(
            f"[INFO] Existing output found: "
            f"{len(existing_df)} generations"
        )

        df = df[~df["uid"].isin(completed_uids)].copy()

        print(
            f"[INFO] Remaining prompts to generate: "
            f"{len(df)}"
        )

        results = existing_df.to_dict("records")

    else:

        print("[INFO] No previous output found.")

        results = []

    if len(df) == 0:

        print("[SUCCESS] All prompts have already been generated.")

        return

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("[INFO] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("[INFO] Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE
    )

    model = model.to(DEVICE)
    model.eval()

    print("[INFO] Model loaded successfully.")

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    total = len(df)

    for count, (_, row) in enumerate(df.iterrows(), start=1):

        prompt = row["prompt"]

        print(f"[INFO] Generating {count}/{total}")

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )["input_ids"]

        # Move input to GPU if available
        inputs = inputs.to(DEVICE)

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with torch.no_grad():

            output = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True
            )

        # Only keep newly generated tokens
        generated_tokens = output[0][inputs.shape[-1]:]

        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "subset": row["subset"],
                "prompt": prompt,
                "generated_answer": answer,
                "model": MODEL_NAME,
                "seed": SEED,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if count % SAVE_EVERY == 0:

            checkpoint_df = pd.DataFrame(results)

            checkpoint_df.to_parquet(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"[CHECKPOINT] Saved {len(results)} "
                f"generations."
            )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    result_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("[SUCCESS] Full generation completed.")
    print(f"[INFO] Total generations: {len(result_df)}")
    print(f"[INFO] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()