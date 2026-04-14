import argparse
import base64
import io
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI

from datasets import load_dataset
from anthropic import AnthropicBedrock

from transformers import Gemma3ForConditionalGeneration

import torch
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig


# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_ID = "mgolov/graded-color-attribution"
SPLIT      = "eval"
MODEL_ID   = "Qwen/Qwen3.5-35B-A3B" #"Qwen/Qwen3.5-9B"

# Used for ALL CoT / introspection turns
INTROSPECTION_PROMPT = (
    "Derive a rule to determine the color of a given object in an image. "
    "Reason step by step, then end your response with exactly: final_threshold=x% "
    "where x is the minimum pixel percentage required to assign a color label."
)

# Used for ALL direct-answer turns
QUESTION_PROMPT = (
    "What color is the {object} in the image? "
    "Reply with the format: final_answer=color"
)

COLOR_EST_PROMPT = (
    "What percentage of pixels in the {object} are {color}?"
    "Reply with the format: estimated_percentage=x%"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_eval_dataset(n=None):
    ds = load_dataset(DATASET_ID, split=SPLIT)
    return ds if n is None else ds.select(range(n))

def _is_openai_model(model_id: str) -> bool:
    return model_id.startswith("gpt") or model_id.startswith("o1") or model_id.startswith("o3")

def _pil_to_base64(image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_model(model_id: str, dtype=torch.bfloat16, device_map: str = "auto"):
    n_gpus = torch.cuda.device_count()

    # Use real free memory with a safety margin
    max_memory = {}
    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = (free / 1e9) * 0.85  # 15% safety margin
        max_memory[i] = f"{free_gb:.0f}GiB"
        print(f"[load_model] GPU {i}: {free/1e9:.1f}GB free → allocating {free_gb:.0f}GB")

    max_memory["cpu"] = "64GiB"

    print(f"[load_model] Loading in BF16 with CPU offload for overflow layers")

    if _is_gemma_model(model_id):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        ).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
    
    processor = AutoProcessor.from_pretrained(model_id)

    # Report what ended up where
    gpu_layers = sum(1 for _, p in model.named_parameters() if "cuda" in str(p.device))
    cpu_layers = sum(1 for _, p in model.named_parameters() if p.device.type == "cpu")
    print(f"[load_model] {gpu_layers} params on GPU, {cpu_layers} params on CPU")

    return model, processor

# ── Core generation primitives ────────────────────────────────────────────────
def _generate_hf(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """HuggingFace generation step."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    answer  = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    updated_messages = messages + [{"role": "assistant", "content": answer}]
    return answer, updated_messages

def _generate_openai(
    client: "OpenAI",
    model_id: str,
    messages: list[dict],
    max_new_tokens: int = 2048,
) -> tuple[str, list[dict]]:

    def _convert_messages(msgs):
        converted = []

        for msg in msgs:
            role = msg["role"]
            content = msg["content"]

            # simple text message
            if isinstance(content, str):
                converted.append({
                    "role": role,
                    "content": content
                })
                continue

            blocks = []
            for block in content:
                if block["type"] == "text":
                    blocks.append({
                        "type": "input_text",
                        "text": block["text"]
                    })

                elif block["type"] == "image":
                    b64 = _pil_to_base64(block["image"])
                    blocks.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}"
                    })

            converted.append({
                "role": role,
                "content": blocks
            })

        return converted

    oai_messages = _convert_messages(messages)

    response = client.responses.create(
        model=model_id,
        input=oai_messages,
        max_output_tokens=max_new_tokens,
        reasoning={"effort": "minimal"},  # ensures reasoning isn't hidden
    )

    answer = response.output_text.strip()

    updated_messages = messages + [{"role": "assistant", "content": answer}]

    return answer, updated_messages

def _is_claude_model(model_id: str) -> bool:
    return "claude" in model_id.lower()

def _generate_claude(
    client: "AnthropicBedrock",
    model_id: str,
    messages: list[dict],
    max_new_tokens: int = 512,
) -> tuple[str, list[dict]]:

    def _convert_messages(msgs):
        converted = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            blocks = []
            for block in content:
                if block["type"] == "text":
                    blocks.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    b64 = _pil_to_base64(block["image"])
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    })

            converted.append({"role": role, "content": blocks})
        return converted

    claude_messages = _convert_messages(messages)

    response = client.messages.create(
        model=model_id,
        thinking={"type": "disabled"},
        max_tokens=max_new_tokens,
        messages=claude_messages,
    )

    answer = response.content[0].text.strip()
    updated_messages = messages + [{"role": "assistant", "content": answer}]
    return answer, updated_messages

def _is_llama_vision_model(model_id: str) -> bool:
    return "llama" in model_id.lower()


def _generate_hf_llama(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """HuggingFace generation for LLaMA 3.2 Vision."""

    # Convert your internal format to LLaMA's expected format
    # and extract PIL images in order
    converted = []
    images = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        blocks = []
        for block in content:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                blocks.append({"type": "image"})  # LLaMA uses a placeholder
                images.append(block["image"])      # PIL image collected separately

        converted.append({"role": role, "content": blocks})

    text = processor.apply_chat_template(
        converted,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    answer  = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    updated_messages = messages + [{"role": "assistant", "content": answer}]
    return answer, updated_messages


def _is_gemma_model(model_id: str) -> bool:
    return "gemma" in model_id.lower()

def _generate_hf_gemma(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """HuggingFace generation for Gemma 3 Vision."""

    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # FIX 2: Gemma's apply_chat_template requires content to always be a
        # list of blocks, never a bare string — even for text-only turns.
        # Passing a string causes the template to iterate over characters and
        # then crash on char["type"] with "string indices must be integers".
        if isinstance(content, str):
            converted.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
            continue

        blocks = []
        for block in content:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                blocks.append({"type": "image", "image": block["image"]})
        converted.append({"role": role, "content": blocks})

    inputs = processor.apply_chat_template(
        converted,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Cast only float tensors to bfloat16; leave input_ids / attention_mask as int64.
    inputs = {
        k: (v.to(model.device, dtype=torch.bfloat16)
            if v.is_floating_point()
            else v.to(model.device))
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # FIX 1: During generate(), Gemma expands image placeholder tokens into
    # hundreds of patch tokens (id=262144), making the output much longer than
    # input_len suggests. Trim past the last image patch token to find where
    # the actual generated text begins.
    GEMMA_IMAGE_TOKEN_ID = 262144
    out = output_ids[0]
    image_positions = (out == GEMMA_IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(image_positions) > 0:
        true_input_end = image_positions[-1].item() + 1
    else:
        true_input_end = input_len  # text-only fallback

    trimmed = out[true_input_end:]
    answer = processor.decode(trimmed, skip_special_tokens=True).strip()

    updated_messages = messages + [{"role": "assistant", "content": answer}]
    return answer, updated_messages

def _make_generate(model_id, model_or_client, processor_or_none):
    if _is_openai_model(model_id):
        client = model_or_client
        def _generate(messages, max_new_tokens=2048):
            return _generate_openai(client, model_id, messages, max_new_tokens)
    elif _is_claude_model(model_id):
        client = model_or_client
        def _generate(messages, max_new_tokens=2048):
            return _generate_claude(client, model_id, messages, max_new_tokens)
    elif _is_llama_vision_model(model_id):
        model, processor = model_or_client, processor_or_none
        def _generate(messages, max_new_tokens=2048):
            return _generate_hf_llama(model, processor, messages, max_new_tokens)

    elif _is_gemma_model(model_id):
        model, processor = model_or_client, processor_or_none
        def _generate(messages, max_new_tokens=2048):
            return _generate_hf_gemma(model, processor, messages, max_new_tokens)
    else:
        model, processor = model_or_client, processor_or_none
        def _generate(messages, max_new_tokens=2048):
            return _generate_hf(model, processor, messages, max_new_tokens)
    return _generate

# ── Experiment 0: Estimate color percentages ─────────────────────────────────────
def estimate_color_percentage(generate, row, max_new_tokens=512):
    """Estimate color thresholds."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": COLOR_EST_PROMPT.format(object=row["object"], color=row['color'])},
            ],
        }
    ]
    answer, _ = generate(messages, max_new_tokens)
    return {"mode": "color_estimation", "answer": answer}

    
# ── Experiment 1: IMG + CoT (single shot) ─────────────────────────────────────   
    
def exp_img_cot_prompt(generate, row, max_new_tokens=2048):
    """Image + introspection prompt in one turn."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": INTROSPECTION_PROMPT + ' ' + QUESTION_PROMPT.format(object=row["object"])},
            ],
        }
    ]
    answer, _ = generate(messages, max_new_tokens)
    return {"mode": "img_cot_prompt", "answer": answer}


# ── Experiment 2: CoT (text only) → IMG + Question (multi-turn) ───────────────
def exp_cot_then_img(generate, row, max_new_tokens=2048):
    """
    Turn 1 (text only): ask model to derive a color-attribution rule via CoT.
    Turn 2 (image):     answer the direct question given the rule just derived.
    """
    messages_t1 = [{"role": "user", "content": INTROSPECTION_PROMPT}]
    cot_answer, messages_after_t1 = generate(messages_t1, max_new_tokens)

    question    = QUESTION_PROMPT.format(object=row["object"])
    messages_t2 = messages_after_t1 + [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": question},
            ],
        }
    ]
    final_answer, _ = generate(messages_t2, max_new_tokens)

    return {"mode": "cot_then_img", "cot_answer": cot_answer, "answer": final_answer}


# ── Experiment 3: CoT + IMG → Question (multi-turn) ───────────────────────────
def exp_cot_img_then_prompt(generate, row, max_new_tokens=2048):
    """
    Turn 1 (image + CoT): show image and ask model to derive a rule.
    Turn 2 (text only):   ask the direct color question.
    """
    messages_t1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": INTROSPECTION_PROMPT},
            ],
        }
    ]
    cot_answer, messages_after_t1 = generate(messages_t1, max_new_tokens)

    question    = QUESTION_PROMPT.format(object=row["object"])
    messages_t2 = messages_after_t1 + [{"role": "user", "content": question}]
    final_answer, _ = generate(messages_t2, max_new_tokens)

    return {"mode": "cot_img_then_prompt", "cot_answer": cot_answer, "answer": final_answer}


# ── Experiment 4: IMG + Question → CoT introspection (multi-turn) ─────────────
def exp_img_prompt_then_cot(generate, row, max_new_tokens=2048):
    """
    Turn 1 (image + question): direct answer first.
    Turn 2 (text only):        ask for CoT introspection / rule derivation.
    """
    question    = QUESTION_PROMPT.format(object=row["object"])
    messages_t1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text",  "text": question},
            ],
        }
    ]
    initial_answer, messages_after_t1 = generate(messages_t1, max_new_tokens)

    messages_t2 = messages_after_t1 + [{"role": "user", "content": INTROSPECTION_PROMPT}]
    cot_answer,  _ = generate(messages_t2, max_new_tokens)

    return {"mode": "img_prompt_then_cot", "initial_answer": initial_answer, "answer": cot_answer}


# ── Dispatcher ────────────────────────────────────────────────────────────────
EXPERIMENT_FNS = {
    # "img_cot_prompt":      exp_img_cot_prompt,
    "color_estimates":     estimate_color_percentage, 
    # "cot_then_img":        exp_cot_then_img,
    # "cot_img_then_prompt": exp_cot_img_then_prompt,
    # "img_prompt_then_cot": exp_img_prompt_then_cot,
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run color attribution experiments.")
    parser.add_argument("--model_id",       type=str, required=True,         help="HuggingFace model ID or OpenAI model name (e.g. gpt-5) or Claude")
    parser.add_argument("--output_path",    type=str, default="results.csv", help="Path to save results CSV")
    parser.add_argument("--max_new_tokens", type=int, default=1024,          help="Max new tokens per generation")
    parser.add_argument("--index_range",    type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Index range to process, e.g. --index_range 300 500 (end is exclusive)")
    args = parser.parse_args()
    
    dataset = load_eval_dataset()
    dataset = dataset.filter(lambda x: x['variant_region'] == 'foreground')
    print(f"Filtered dataset size: {len(dataset)}")
    
    if args.index_range is not None:
        start, end = args.index_range
        assert start < len(dataset), f"start={start} exceeds dataset size {len(dataset)}"
        indices = list(range(start, min(end, len(dataset))))
    else:
        indices = list(range(len(dataset)))
    
    print(f"Processing {len(indices)} samples (idx {indices[0]}–{indices[-1]})")
    
    # ── Load the right backend ────────────────────────────────────────────────
    if _is_openai_model(args.model_id):
        api_key   = "INSERT_KEY_HERE" 
        client    = OpenAI(api_key=api_key)
        processor = None
        generate  = _make_generate(args.model_id, client, processor)
        print(f"Using OpenAI backend: {args.model_id}")
    
    elif _is_claude_model(args.model_id):

        client = AnthropicBedrock(
            aws_access_key="INSERT_KEY_HERE",
            aws_secret_key="INSERT_KEY_HERE",
            aws_region="INSERT_REGION_HERE",
        )
        processor = None
        generate  = _make_generate(args.model_id, client, processor)
        print(f"Using Claude/Bedrock backend: {args.model_id}")
        
    else:
        model, processor = load_model(model_id=args.model_id)
        generate         = _make_generate(args.model_id, model, processor)
        print(f"Using HuggingFace backend: {args.model_id}")
    
    # ── Run experiments ───────────────────────────────────────────────────────
    records = []
    for idx in tqdm(indices, desc="Samples"):
        row = dataset[idx]
        for mode, fn in EXPERIMENT_FNS.items():
            try:
                r = fn(generate, row, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"[ERROR] idx={idx} mode={mode}: {e}")
                r = {"answer": None, "cot_answer": None, "initial_answer": None}
            records.append({
                "dataset_index":   idx,
                "mode":            mode,
                "model":           args.model_id,
                "object":          row["object"],
                "gt_color":        row["color"],
                "stimulus_type":   row["stimulus_type"],
                "percent_colored": row["percent_colored"],
                "subset":          row["subset"],
                "cot_answer":      r.get("cot_answer"),
                "initial_answer":  r.get("initial_answer"),
                "answer":          r.get("answer"),
            })
    
        if idx % 100 == 0:
            sub_df = pd.DataFrame(records)
            safe_model_id = args.model_id.replace("/", "_")
            print(f"SAVING DF AT IDX: {idx}")
            sub_df.to_csv(f'{safe_model_id}_results_idx_{idx}_color_estimates.csv', index=False)
                
    df = pd.DataFrame(records)
    safe_model_id = args.model_id.replace("/", "_")
    df.to_csv(f'{safe_model_id}_results_color_estimates.csv', index=False)
    print(f"Saved {len(df)} rows.")


if __name__ == "__main__":
    main()