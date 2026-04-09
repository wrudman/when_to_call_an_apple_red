"""
model_priors.py

Module for generating and analyzing model-based color priors for visual-language models (VLMs),
for open-weight models e.g. LLaVA, Qwen in ModelColorPriors class, and for GPT-Vision models in GPTColorPriors class.
"""

import gc
import ast
import torch
import pandas as pd
from tqdm import tqdm
import base64
from pathlib import Path
import os
from openai import OpenAI
from PIL import Image
from model_evaluation.test_MLLMs import prompt_mllm
from collections import Counter
import string


class BaseColorPriors:
    """
    Abstract parent class for color prior extraction.
    Contains all shared logic and utilities.
    """

    def __init__(self, model_name, data_folder):
        self.model_name = model_name
        self.data_folder = data_folder

    #  Abstract methods to override
    def query_model_dummy(self, df, question):
        """Return a color string (text-only)."""
        raise NotImplementedError

    def query_model_image(self, df, question, image_path):
        """Return a color string (image + question)."""
        raise NotImplementedError
    

    def create_prior_prompt(self, object_name, most=True, use_image=False):
        """
        Create a prompt asking for up to 3 likely colors.

        If use_image=False: ask about object category (pure priors)
        If use_image=True: force GPT to use visual clues in THIS SPECIFIC image

        Returns a LLaVA-style prompt with [INST] <image> ... [/INST]
        if use_image=True, otherwise no <image> token.
        """

        # ----------------------------------------------------
        # 1. Optional <image> token wrapper (for open-weight VLMs)
        # ----------------------------------------------------
        if use_image:
            instruction_tokens = "[INST] <image>\n"
        else:
            instruction_tokens = "[INST]\n"

        end_tokens = "[/INST]"

        # ----------------------------------------------------
        # 2. Build the *task* part of the question
        # ----------------------------------------------------
        if not use_image:
            # --- PRIOR MODE (no image used) ---
            if most == "True":
                plural = object_name if object_name.endswith("s") else object_name + "s"
                question = (
                    f"List up to three possible colors for most {plural}, "
                    f"from most likely to least likely."
                )
            else:
                question = (
                    f"List up to three possible colors for this {object_name}, "
                    f"from most likely to least likely."
                )

        else:
            # --- IMAGE-CONDITIONED MODE ---
            # Forces GPT to *LOOK* at the grayscale outline
            question = (
                "Based ONLY on the visual appearance, structure, and clues in THIS grayscale image, "
                "list up to three plausible real-world colors this specific object might have. "
                "Do NOT rely on world knowledge unless the silhouette clearly indicates an object type."
            )

        # ----------------------------------------------------
        # 3. Formatting rule
        # ----------------------------------------------------
        format_rule = (
            "Respond ONLY with English color words separated by commas. "
            "Do not use explanations or sentences."
        )

        # ----------------------------------------------------
        # 4. Assemble prompt
        # ----------------------------------------------------
        prompt = (
            f"{instruction_tokens}"
            f"{question} {format_rule}"
            f" {end_tokens}"
        )

        return prompt

        

    def get_model_color_priors(self, df, most=True, save=True):
        """
        Shared implementation:
        Calls subclass methods for priors with and without image.
        """
        results = []
        batch_size = 1  # larger batch sizes not implemented yet

        def _parse_colors(s):
            """Convert 'red, green, blue' to ['red','green','blue']"""
            parts = [p.strip() for p in s.split(",") if p.strip()]
            
            return parts


        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Color priors ({self.model_name})"):
            batch_df = df.iloc[i : i + batch_size].reset_index(drop=True)
            object_name = row["object"]
            prompt = self.create_prior_prompt(object_name, most=str(most))

            dummy_priors = self.query_model_dummy(batch_df, prompt)
            img_priors  = self.query_model_image(batch_df, prompt)

            out_batch = {
                "object": row["object"],
                "correct_answer": row["correct_answer"],
                "dummy_priors": _parse_colors(dummy_priors[0]),
                "image_priors": _parse_colors(img_priors[0]),
            }

            results.append(out_batch)
            del out_batch
            del batch_df
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        ground_truth_df = pd.DataFrame(results)       

        if save:
            out_path = self.data_folder / f"color_priors_{self.model_name}.csv"
            ground_truth_df.to_csv(out_path, index=False)
            print("Saved priors to: ", out_path)

        return ground_truth_df

    def pick_primary_color(self, df, column="dummy_priors", allow_black=False):
        """
        Select the best (primary) color for each row:
            • drop excluded colors
            • drop colors not in allowed vocabulary
            • if none remain, return None
            • map silver to grey, gold to yellow
            • return a flag for rows where a change happened
        """

        EXCLUDE = {"white", "silver", "gold", "clear"}
        if not allow_black:
            EXCLUDE.add("black")

        ALLOWED = {
            "red", "brown", "pink", "orange", "yellow", "gold",
            "green", "blue", "purple", "black", "grey", "silver", "white"
        }

        # Final mapping AFTER selection
        COLOR_MAP = {
            "silver": "grey",
            "gold": "yellow"
        }

        primary_colors = []
        changed_flags = []

        for obj, priors in zip(df["object"], df[column]):

            changed = False  # track modifications

            # Ensure list
            if not isinstance(priors, list):
                print(f"[WARN] priors for {obj} not a list: {priors}")
                primary_colors.append(None)
                changed_flags.append(True)
                continue

            # Normalize
            priors = [str(c).lower().strip() for c in priors]
            original_first = priors[0] if priors else None

            # Filter invalid colors
            filtered = [
                c for c in priors
                if c in ALLOWED and c not in EXCLUDE
            ]

            if len(filtered) == 0:
                # if no valid colors remain write None
                print(f"[NULL] {obj}: all priors invalid {priors} to NaN")
                primary_colors.append(None)
                changed_flags.append(True)
                continue

            # First valid color
            chosen = filtered[0]

            # Check if this differs from original first
            if chosen != original_first:
                changed = True
                print(f"[INFO] {obj}: replaced '{original_first}' with '{chosen}'")

            # Apply final color mapping (silver->grey, gold->yellow)
            mapped = COLOR_MAP.get(chosen, chosen)
            if mapped != chosen:
                print(f"[MAP] {obj}: mapped '{chosen}' to '{mapped}'")
                changed = True

            primary_colors.append(mapped)
            changed_flags.append(changed)

        return primary_colors, changed_flags



    # Shared analysis
    def analyze_differences(self, df):
        """
        Add diagnostic columns for dummy vs image priors.
        """
        df = df.copy()

        # Check if model_prior color in GT
        df["prior_in_gt"] = df.apply(
            lambda r: r["prior"].lower()
            in [c.lower() for c in r["correct_answer"]],
            axis=1,
        )
        print(f"{df['prior_in_gt'].sum()} rows where the chosen model color prior is NOT in ground truth from Visual Counterfact.")

        model_priors = df["prior"].unique()
        print(f"Model color priors: {model_priors}")


    def load_model_priors(self):
        path = self.data_folder / f"color_priors_{self.model_name}.csv"
        df = pd.read_csv(path)
        df["correct_answer"] = df["correct_answer"].apply(ast.literal_eval)
        df["dummy_priors"] = df["dummy_priors"].apply(ast.literal_eval)
        df["image_priors"] = df["image_priors"].apply(ast.literal_eval)
        return df


class TorchColorPriors(BaseColorPriors):
    """
    Color prior extraction using open-weight VLMs (with processor and torch model).
    """
    def __init__(self, processor, model, device, data_folder):
        super().__init__(model.name_or_path.split("/")[-1], data_folder)
        self.processor = processor
        self.model = model
        self.device = device

    def query_model_dummy(self, df, prompt):
        result = prompt_mllm(
            df,
            self.processor,
            self.model,
            self.device,
            prompt=prompt,
            dummy=True
        )
        return result["predicted_color"].tolist()

    def query_model_image(self, df, prompt):
        result = prompt_mllm(
            df,
            self.processor,
            self.model,
            self.device,
            prompt=prompt,
            dummy=False
        )
        return result["predicted_color"].tolist()


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class GPTColorPriors(BaseColorPriors):
    """
    Color-prior extraction using GPT-Vision models (no processor, no torch model).
    Mirrors the behavior of TorchColorPriors.
    """

    def __init__(self, model_name, data_folder):
        super().__init__(model_name, data_folder)
        API_KEY = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=API_KEY)

   
    def ask_gpt_raw(self, prompt: str, image_path: str | None = None):

        if image_path is None:
            # text only
            messages = [
                {"role": "system", "content": "You are a strict JSON generator for image analysis tasks."},
                {"role": "user", "content": prompt}
            ]



            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                top_p=0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()

        else:
        # image + text
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
           
            messages = [
                {"role": "system", "content": "You are a strict JSON generator for image analysis tasks."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                top_p=0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()

    
    # Parse answer list from GPT
    def parse_prior_list(self, text: str):
        if text is None:
            return []

        # Normalize punctuation spacing
        text = text.lower().replace("gray", "grey").strip()

        # Split on commas
        items = text.split(",")

        cleaned = []
        for item in items:
            token = item.strip()

            # Remove trailing punctuation: ".", ",", "!", "?", ";", ":"
            token = token.strip(string.punctuation)

            if token:
                cleaned.append(token)

        return cleaned


    # Query color prior without image (text only)
    def query_model_dummy(self, prompt: str):
        raw = self.ask_gpt_raw(prompt, image_path=None)
        return self.parse_prior_list(raw)
    

    # Query color prior with image
    def query_model_image(self, prompt: str, image_path: str):
        raw = self.ask_gpt_raw(prompt, image_path=image_path)
        return self.parse_prior_list(raw)

  
    # Main color prior extraction function
    def get_model_color_priors(self, df, most=True, save=True):
        rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GPT Priors ({self.model_name})"):
            obj = row["object"]
            img = row["image_path"]

            prompt_dummy = self.create_prior_prompt(obj, most=str(most))
            prompt_image = self.create_prior_prompt(obj, most=str(most), use_image=True)

            prior_dummy = self.query_model_dummy(prompt_dummy)
            prior_image = self.query_model_image(prompt_image, img)

            rows.append({
                "object": obj,
                "correct_answer": row["correct_answer"],
                "dummy_priors": prior_dummy,
                "image_priors": prior_image,
            })

        priors_df = pd.DataFrame(rows)

        if save:
            out_path = self.data_folder / f"color_priors_{self.model_name}.csv"
            priors_df.to_csv(out_path, index=False)
            print(f"Saved GPT color priors to {out_path}")

        return priors_df