from transformers import (
    BitsAndBytesConfig,LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from PIL import Image
import torch
import pandas as pd
import re
import argparse
from tqdm import tqdm
import os
import io
import gc
import base64
from pathlib import Path
import torch.nn.functional as F
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import asyncio

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()
client = OpenAI()
client_async = AsyncOpenAI()


def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def create_eval_prompt(
    object_name: str,
    *,
    most: bool = False,
    calibration_value: int | None = None,
    multiturn: bool = False,
):
    """
    Builds either a normal or calibrated prompt.
    """

    intro = ""
    if calibration_value is not None and not multiturn:
        intro = (
            f"For any object, {calibration_value}% of its pixels should be colored for it to be "
            "considered that color. "
            "Please keep this threshold in mind when answering the next question. "
        )

    if most:
        obj = object_name if object_name.endswith("s") else object_name + "s"
        question = f"What color are most {obj}?"
    else:
        question = f"What color is this {object_name}?"

    prompt = (
        f"{intro}Answer with one word. {question}"
    )
    return prompt


def create_multiturn_eval_prompt(
    object_name: str,
    *,
    most: bool = False,
):
    if most:
        obj = object_name if object_name.endswith("s") else object_name + "s"
        question = f"What color are most {obj}?"
    else:
        question = f"What color is this {object_name}?"

    return f"Answer with one word. {question}"



def prompt_mllm(df, processor, model, device, prompt, dummy=False, top_k=5, return_probs=False):

    df = df.copy()
    df["pred_color_this"] = None
    df["logprob_pred_token"] = None
    df["logprob_correct_token"] = None
    df["correct_in_top_k"] = False

    with torch.inference_mode():

        for idx, row in df.iterrows():

            if dummy:
                inputs = processor(text=prompt, return_tensors="pt")
            else:
                try:
                    image = Image.open(row["image_path"]).convert("RGB")
                    inputs = processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt"
                    )
                except FileNotFoundError:
                    continue

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            # --- Remove prompt tokens ---
            generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]

            decoded = processor.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip().lower()

            pred = decoded.replace("gray", "grey").split()[0]
            df.at[idx, "pred_color_this"] = pred

            # --- Logprob extraction ---
            if len(outputs.scores) > 0:

                first_step_logits = outputs.scores[0]  # (batch, vocab)
                logprobs = F.log_softmax(first_step_logits, dim=-1)

                pred_token_id = generated_ids[0, 0]
                pred_logprob = logprobs[0, pred_token_id].item()

                df.at[idx, "logprob_pred_token"] = pred_logprob

                # --- Correct token logprob ---
                correct = row["correct_answer"].lower()

                correct_ids = processor.tokenizer(
                    correct,
                    add_special_tokens=False
                )["input_ids"]

                if len(correct_ids) == 1:
                    correct_id = correct_ids[0]
                    correct_lp = logprobs[0, correct_id].item()

                    df.at[idx, "logprob_correct_token"] = correct_lp

                    # --- Top-k inclusion ---
                    topk_ids = torch.topk(
                        logprobs[0], k=top_k
                    ).indices.tolist()

                    df.at[idx, "correct_in_top_k"] = (
                        correct_id in topk_ids
                    )

            # cleanup
            del inputs, outputs
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return df


def prompt_qwen(df, processor, model, device, prompt, top_k=5):

    df = df.copy()
    df["pred_color_this"] = None
    df["logprob_pred_token"] = None
    df["logprob_correct_token"] = None
    df["correct_in_top_k"] = False

    for idx, row in df.iterrows():

        image = Image.open(row["image_path"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # --- Remove prompt tokens ---
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip().lower()

        pred = decoded.replace("gray", "grey").split()[0]
        df.at[idx, "pred_color_this"] = pred

        # --- Logprob extraction ---
        if len(outputs.scores) > 0:
            first_step_logits = outputs.scores[0]  # (batch, vocab)
            logprobs = F.log_softmax(first_step_logits, dim=-1)

            pred_token_id = generated_ids[0, 0]
            pred_logprob = logprobs[0, pred_token_id].item()

            df.at[idx, "logprob_pred_token"] = pred_logprob

            # Check correct token
            correct = row["correct_answer"].lower()

            correct_ids = processor.tokenizer(
                correct,
                add_special_tokens=False
            )["input_ids"]

            if len(correct_ids) == 1:
                correct_id = correct_ids[0]
                correct_lp = logprobs[0, correct_id].item()
                df.at[idx, "logprob_correct_token"] = correct_lp

                # Top-k inclusion
                topk_ids = torch.topk(logprobs[0], k=top_k).indices.tolist()
                df.at[idx, "correct_in_top_k"] = correct_id in topk_ids

    return df



def encode_image_to_b64(path):
    """Return base64 string for a local image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    


def prompt_gpt(
    df,
    prompt,
    model_name="gpt-4o",
    dummy=False,
    top_k=5,
):
    df = df.copy()

    # Pre-create output columns
    df["pred_color_this"] = None
    df["logprob_pred_token"] = None
    df["logprob_correct_token"] = None
    df["correct_in_top_k"] = False
    df["prob_correct_this"] = None

    for idx, row in df.iterrows():

        # Build image
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            try:
                img_b64 = encode_image_to_b64(row["image_path"])
            except FileNotFoundError:
                continue  # leave row as None

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                max_tokens=10,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_k,
            )

            choice = response.choices[0]

            # Predicted text
            ans = choice.message.content.strip().lower()
            ans = ans.replace("gray", "grey").split()[0]

            df.at[idx, "pred_color_this"] = ans

            # Logprobs
            token_info = choice.logprobs.content

            if token_info and len(token_info) > 0:
                first_token = token_info[0]

                df.at[idx, "logprob_pred_token"] = first_token.logprob

                correct = str(row["correct_answer"]).lower()
                found = False
                lp_correct = None

                for cand in first_token.top_logprobs:
                    tok = cand.token.lower().replace("gray", "grey")
                    if tok == correct:
                        found = True
                        lp_correct = cand.logprob
                        break

                df.at[idx, "correct_in_top_k"] = found
                df.at[idx, "logprob_correct_token"] = lp_correct


        except Exception as e:
            print("GPT error:", e)
            continue

    return df


def prompt_gpt_multiturn(
    df,
    prompt,
    introspection_threshold,
    model_name="gpt-4o",
    top_k=5,
):
    df = df.copy()

    df["pred_color_this"] = None
    df["logprob_pred_token"] = None
    df["logprob_correct_token"] = None
    df["correct_in_top_k"] = False

    for idx, row in df.iterrows():

        # Image
        try:
            img_b64 = encode_image_to_b64(row["image_path"])
        except FileNotFoundError:
            continue

        # Multi-turn message history
        messages = [
            {
                "role": "user",
                "content": INTROSPECTION_PROMPT,
            },
            {
                "role": "assistant",
                "content": str(introspection_threshold),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    },
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_k,
            )

            choice = response.choices[0]

            ans = choice.message.content.strip().lower()
            ans = ans.replace("gray", "grey").split()[0]

            df.at[idx, "pred_color_this"] = ans

            token_info = choice.logprobs.content

            if token_info and len(token_info) > 0:
                first_token = token_info[0]
                df.at[idx, "logprob_pred_token"] = first_token.logprob

                correct = str(row["correct_answer"]).lower()
                found = False
                lp_correct = None

                for cand in first_token.top_logprobs:
                    tok = cand.token.lower().replace("gray", "grey")
                    if tok == correct:
                        found = True
                        lp_correct = cand.logprob
                        break

                df.at[idx, "correct_in_top_k"] = found
                df.at[idx, "logprob_correct_token"] = lp_correct

        except Exception as e:
            print("GPT error:", e)
            continue

    return df


def prompt_gpt52(
    df,
    prompt,
    model_name="gpt-5.2",
    dummy=False,
    top_k=5,
):
    preds = []
    logprob_preds = []
    logprob_corrects = []
    correct_in_topk = []

    for _, row in df.iterrows():

        # image handling
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            img_b64 = encode_image_to_b64(row["image_path"])

        # GPT-5.2 request
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{img_b64}",
                        },
                    ],
                }
            ],
            max_output_tokens=16,
            temperature=0.0,
            output=[
                {
                    "type": "output_text",
                    "logprobs": {
                        "top_k": top_k
                    },
                }
            ],
        )

        # extract output_text
        text_items = [
            item for item in response.output
            if item["type"] == "output_text"
        ]

        if not text_items:
            preds.append(None)
            logprob_preds.append(None)
            logprob_corrects.append(None)
            correct_in_topk.append(False)
            continue

        out = text_items[0]
        tokens = out.get("tokens", [])

        if not tokens:
            preds.append(None)
            logprob_preds.append(None)
            logprob_corrects.append(None)
            correct_in_topk.append(False)
            continue

        # predicted token
        pred_token = (
            tokens[0]["token"]
            .lower()
            .replace("gray", "grey")
        )
        preds.append(pred_token)

        logprob_preds.append(tokens[0]["logprob"])

        # correct color handling
        correct = str(row["correct_answer"]).lower()
        found = False
        lp_correct = None

        for cand in tokens[0].get("top_logprobs", []):
            tok = cand["token"].lower().replace("gray", "grey")
            if tok == correct:
                found = True
                lp_correct = cand["logprob"]
                break

        correct_in_topk.append(found)
        logprob_corrects.append(lp_correct)

    df = df.copy()
    df["pred_color_this"] = preds
    df["logprob_pred_token"] = logprob_preds
    df["logprob_correct_token"] = logprob_corrects
    df["correct_in_top_k"] = correct_in_topk

    return df


async def prompt_gpt_async(df, prompt, model_name="gpt-4o", dummy=False):
    """
    Async GPT equivalent of prompt_gpt().
    Returns df with a new column: pred_color_this.
    All requests are sent concurrently.
    """

    async def query_single(row):
        # Build / encode image
        if dummy:
            img = Image.new("RGB", (512, 512), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            try:
                img_b64 = encode_image_to_b64(row["image_path"])
            except FileNotFoundError:
                return None

        try:
            response = await client_async.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0.0,
                top_p=0,
            )

            ans = response.choices[0].message.content.strip().lower()
            return ans.split()[0].replace("gray", "grey")

        except Exception as e:
            print("GPT error:", e)
            return None

    # Build tasks (no batching)
    tasks = [query_single(row) for _, row in df.iterrows()]

    # Run in parallel
    preds = await asyncio.gather(*tasks)

    df = df.copy()
    df["pred_color_this"] = preds
    return df


def run_async(coro):
    """
    Safe asyncio runner: 
    - uses asyncio.run() when no loop is running
    - uses await via nest_asyncio when in Jupyter
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside Jupyter
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        # Normal Python script
        return asyncio.run(coro)
    

def prompt_gpt_sync(df, prompt, model_name="gpt-4o", dummy=False):
    return run_async(prompt_gpt_async(df, prompt, model_name=model_name, dummy=dummy))


def run_vlm_evaluation(
    df,
    *,
    backend: str,                    # "llava" | "qwen" | "gpt4" | "gpt52"
    processor=None,
    model=None,
    device=None,
    model_name=None,
    calibration_value: int | None = None,
    mode="this",
    multiturn_introspection=False,
):
    """
    Generic evaluation loop for Vision-Language Models.

    backend:
        - "llava": open-weight torch VLMs (LLaVA, etc.)
        - "gpt":   OpenAI GPT models (default: gpt-5o)
        - "qwen":  Qwen-VL models (HF)
    """
     
    results = []

    for _, row in df.iterrows():
        prompt = create_eval_prompt(
            row["object"],
            most=(mode == "most"),
            calibration_value=calibration_value,
            multiturn=multiturn_introspection,
        )

        if backend == "llava":
            prompt = f"[INST] <image>\n{prompt}\n[/INST]"
            out = prompt_mllm(
                df, processor, model, device, prompt, return_probs=True
            )

        elif backend == "qwen":
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            out = prompt_qwen(
                df, processor, model, device, prompt
            )

        elif backend == "gpt4":
            if multiturn_introspection and calibration_value is not None:
                #print(f"INTROSPECTION_PROMPT: {INTROSPECTION_PROMPT}")
                #print(f"prompt: {prompt}")
                out = prompt_gpt_multiturn(
                    df,
                    prompt,
                    introspection_threshold=calibration_value,
                    model_name="gpt-4o",
                )
            else:
                print("single-turn")
                out = prompt_gpt(
                    df,
                    prompt,
                    model_name="gpt-4o",
                )

        elif backend == "gpt52":
            out = prompt_gpt52(
                df, prompt, model_name="gpt-5.2"
            )

        else:
            raise ValueError(backend)

        out["calibration"] = calibration_value
        results.append(out)

    return pd.concat(results, ignore_index=True)


# Helpers for introspection prompt
INTROSPECTION_PROMPT = """For any object, x% of its pixels should be colored for it to be considered that color.
For example, imagine an image of a banana, where only part of the banana in the image is colored yellow.
At what point would you personally say that the banana in the image is yellow?
What value should x% be?
Please only answer with a single number between 0 and 100."""



def parse_percentage(text: str | None) -> int | None:
    if not text:
        return None
    matches = re.findall(r"\b(\d{1,3})\b", text)
    if not matches:
        return None
    value = int(matches[-1])   # ← take LAST number
    return value if 0 <= value <= 100 else None



DUMMY_IMAGE = Image.new("RGB", (512, 512), "white")

def ask_vlm_introspection_threshold(
    *,
    backend: str,                  # "llava" | "qwen" | "gpt"
    processor=None,
    model=None,
    device=None,
    model_name: str | None = None,
) -> dict:

    if backend == "llava":
        prompt = f"[INST] <image>\n{INTROSPECTION_PROMPT}\n[/INST]"

        inputs = processor(
            #images=DUMMY_IMAGE,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        raw = processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print(raw)
        raw = clean_instruction_tokens(raw).strip().lower()

    elif backend == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INTROSPECTION_PROMPT},
                ],
            }
        ]

        # build the chat-formatted text 
        chat_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        # Stokenize / preprocess into tensors
        inputs = processor(
            #images=DUMMY_IMAGE,
            text=chat_text,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        raw = processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip().lower()


    elif backend == "gpt":
        gpt_model = model_name or "gpt-5.2"

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": INTROSPECTION_PROMPT,
                }
            ],
            temperature=0.0,
            max_completion_tokens=50,
        )
        raw = response.choices[0].message.content.strip().lower()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    threshold = parse_percentage(raw)

    return {
        "backend": backend,
        "model_name": model_name,
        "introspection_raw": raw,
        "introspection_threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next'], required=True, help="Choose the model version.")
    # NOTE: all images now have a perspective line. Keeping in to not mess up file names. 
    #parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=False, help="Choose dataset size (mini or full).")
    #parser.add_argument('--image_type', type=str, choices=['color', 'grayscale'], required=True, help="Choose color or grayscale image.")
    parser.add_argument("--mode", type=str, choices=["this", "most", "both"], default="both", help="Evaluation mode: 'this', 'most', or 'both'.")
    parser.add_argument("--dummy_image", action="store_true", help="Use a dummy white image instead of the real one (for model priors).", default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set a specific seed for reproducibility
    SEED = 42
    # Setting the seed for PyTorch
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if args.model_version == 'llava-next':
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", quantization_config=bnb_config
        ).to(device)
                
    project_root = Path.cwd()
    data_folder = project_root / "data" / "fruit"

    # Dataset path
    dataset_path = (
        data_folder / "fruit_images.parquet"
    )

    df = pd.read_parquet(dataset_path)
        
    print(f"Running evaluation: mode={args.mode}, dummy={args.dummy_image}")
    df_results = run_vlm_evaluation(
        df=df,
        processor=processor,
        model=model,
        device=device,
        batch_size=1,
        mode=args.mode,
        dummy=args.dummy_image,
    )
    
    out_name = (
        f"results_{args.model_version}_{args.mode}.csv"
    )
    out_path = data_folder / out_name

    df_results.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")

    
if __name__ == "__main__":
    main()