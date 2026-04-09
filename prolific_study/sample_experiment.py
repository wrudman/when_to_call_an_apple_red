import random
import json
from pathlib import Path
import pandas as pd
import re


# CONFIG

SANITY_POSITIONS = [5, 25, 45, 65, 85]  # 1-based indexing
TOTAL_QUESTIONS = 106

PCTS_13 = [0, 5, 10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100]
PCTS_12_NO_ZERO = [5, 10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100]
PCTS_5_BG = [20, 40, 60, 80, 100]


# Object name normalization
COLOR_ADJECTIVES = {
    "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "black", "grey"
}

SPECIAL_RENAMES = {
    "mouse (computer)": "computer mouse",
    "joystick": "game controller",
}


def normalize_object_name(name: str) -> str:
    """
    Normalize object names for participant-facing display:
    - lowercase
    - remove parenthetical disambiguators
    - remove descriptive color adjectives
    """
    if not isinstance(name, str):
        return name
    
    if name in SPECIAL_RENAMES:
        return SPECIAL_RENAMES[name]

    name = name.lower().strip()

    # Remove parentheticals: "bat (animal)" -> "bat"
    name = re.sub(r"\s*\([^)]*\)", "", name)

    tokens = name.split()

    # Remove leading color adjectives (e.g. "green lizard")
    tokens = [t for t in tokens if t not in COLOR_ADJECTIVES]

    return " ".join(tokens)
 

# Helpers
def _sample_unique(df, rng, n, used=None):
    """
    Sample n rows from df, optionally enforcing uniqueness on df["object"].
    """
    if used is not None:
        df = df[~df["object"].isin(used)]

    if len(df) < n:
        raise RuntimeError("Not enough candidates to sample from.")

    sampled = df.sample(n=n, random_state=rng.randint(0, 10**9))
    rows = sampled.to_dict("records")

    if used is not None:
        for r in rows:
            used.add(r["object"])

    return rows



# Object samplers
def sample_counterfactual_objects(df_cf, rng):
    rows = []
    used_objects = set()
    df_cf = df_cf.copy()
    df_cf["percent_colored"] = df_cf["percent_colored"].astype(int)

    for pct in PCTS_12_NO_ZERO:
        candidates = df_cf[df_cf["percent_colored"] == pct]
        rows += _sample_unique(candidates, rng, 1, used_objects)

    assert len(rows) == 12
    return rows


def sample_background_objects(df_bg, rng):
    rows = []
    used_objects = set()
    df_bg = df_bg.copy()
    df_bg["percent_colored"] = df_bg["percent_colored"].astype(int)

    for pct in PCTS_5_BG:
        candidates = df_bg[df_bg["percent_colored"] == pct]
        rows += _sample_unique(candidates, rng, 1, used_objects)

    assert len(rows) == 5
    return rows


def sample_prior_objects(df_priors, rng):
    rows = []
    df_priors = df_priors.copy()
    df_priors["percent_colored"] = df_priors["percent_colored"].astype(int)

    for pct in PCTS_13:
        candidates = df_priors[df_priors["percent_colored"] == pct]
        if candidates.empty:
            raise RuntimeError(f"No candidates found for percent_colored={pct}")
        rows += _sample_unique(candidates, rng, 3)

    assert len(rows) == 39
    return rows



# Shape samplers
def sample_background_shapes(df_bg, rng):
    rows = []
    used_shapes = set()
    df_bg = df_bg.copy()
    df_bg["percent_colored"] = df_bg["percent_colored"].astype(int)

    for pct in PCTS_5_BG:
        candidates = df_bg[df_bg["percent_colored"] == pct]
        rows += _sample_unique(candidates, rng, 1, used_shapes)

    assert len(rows) == 5
    return rows


def sample_priors_shapes(df_priors, rng):
    rows = []
    df_priors = df_priors.copy()
    df_priors["percent_colored"] = df_priors["percent_colored"].astype(int)

    for pct in PCTS_13:
        candidates = df_priors[df_priors["percent_colored"] == pct]
        rows += _sample_unique(candidates, rng, 3)

    assert len(rows) == 39
    return rows


# Fixed questions
def make_introspection_question():
    return {
        "question_type": "introspection",
        "prompt": (
            "<p>"
            "For any object, <b>x%</b> of its pixels should be colored "
            "for it to be considered that color."
            "</p>"
            "<p>"
            "For example, imagine an image of a banana, "
            "where only part of the banana in the image is colored yellow."
            "</p>"
            "<p>"
            "At what point would you personally say that the banana in the image "
            "is yellow?"
            "</p>"
            "<p>"
            "What value should <b>x%</b> be?"
            "</p>"
        ),
        "min": 0,
        "max": 100,
    }


# Sanity questions
SANITY_QUESTIONS = [
    {
        "question_type": "sanity",
        "sanity_id": 1,
        "prompt": 'To show you are paying attention, please select "Strongly Disagree".',
        "options": [
            "Strongly Agree",
            "Agree",
            "Neither Agree nor Disagree",
            "Disagree",
            "Strongly Disagree"
        ],
        "correct_response": "Strongly Disagree",
    },
    {
        "question_type": "sanity",
        "sanity_id": 2,
        "prompt": "Type the word BLUE in the box below.",
        "response_type": "text",
        "correct_response": "BLUE",
    },
    {
        "question_type": "sanity",
        "sanity_id": 3,
        "prompt": "Please select option number 3 for this item.",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "correct_response": "Option 3",
    },
    {
    "question_type": "sanity",
    "sanity_id": 4,
    "prompt": "To show that you are paying attention, please disagree with the following statement.",
    "statement": "I am currently not paying attention.",
    "options": [
        "Agree",
        "Neither agree nor disagree",
        "Disagree"
    ],
    "correct_response": "Disagree"
    },
    {
        "question_type": "sanity",
        "sanity_id": 5,
        "prompt": (
            'For this statement, please select "Neither Agree nor Disagree".'
        ),
        "options": [
            "Strongly Agree",
            "Agree",
            "Neither Agree nor Disagree",
            "Disagree",
            "Strongly Disagree"
        ],
        "correct_response": "Neither Agree nor Disagree",
    },
]

def insert_sanity_questions(questions):
    sanity_positions = [5, 25, 45, 65, 85]  # 1-based
    sanity_iter = iter(SANITY_QUESTIONS)

    for pos in sorted(sanity_positions):
        questions.insert(pos - 1, next(sanity_iter))


# Profile generator
def generate_profile(
    df_priors,
    df_cf,
    df_shapes,
    seed,
    introspection_position,
):
    rng = random.Random(seed)

    questions = []

    # Objects with priors
    df_priors_fg = df_priors[df_priors["variant_region"] == "FG"]
    df_priors_bg = df_priors[df_priors["variant_region"] == "BG"]
    questions += sample_prior_objects(df_priors_fg, rng)
    questions += sample_background_objects(df_priors_bg, rng)
    

    # Counterfacts
    df_cf_fg = df_cf[df_cf["variant_region"] == "FG"]
    questions += sample_counterfactual_objects(df_cf_fg, rng)

    # Shapes
    df_shapes_fg = df_shapes[df_shapes["variant_region"] == "FG"]
    df_shapes_bg = df_shapes[df_shapes["variant_region"] == "BG"]
    questions += sample_priors_shapes(df_shapes_fg, rng)
    questions += sample_background_shapes(df_shapes_bg, rng)

    if len(questions) != 100:
        raise RuntimeError(
            f"Expected 100 variable questions, got {len(questions)}"
        )

    # Randomize all variable questions together
    rng.shuffle(questions)

    # Insert sanity questions
    insert_sanity_questions(questions)

    # Insert introspection
    introspection = make_introspection_question()
    if introspection_position == "first":
        questions.insert(0, introspection)
    elif introspection_position == "last":
        questions.append(introspection)
    else:
        raise ValueError("introspection_position must be 'first' or 'last'")

    if len(questions) != TOTAL_QUESTIONS:
        raise RuntimeError(
            f"Final survey length mismatch: {len(questions)} != {TOTAL_QUESTIONS}"
        )
    
    # Normalize object names for labels
    for q in questions:
        if "object" in q:
            q["object"] = normalize_object_name(q["object"])


    return questions


def generate_debug_profile(df_priors, df_shapes):
    """
    Small deterministic profile for debugging:
    - 5 normal color questions
    - all sanity questions
    - 1 introspection question
    """

    questions = []

    # --- 5 NORMAL QUESTIONS (hand-picked, deterministic) ---

    normals = [
        df_priors.iloc[0].to_dict(),
        df_priors.iloc[1].to_dict(),
        df_shapes.iloc[0].to_dict(),
        df_shapes.iloc[1].to_dict(),
        df_priors.iloc[2].to_dict(),
    ]

    # Ensure required fields exist
    for q in normals:
        q.pop("question_type", None)

    # --- BUILD QUESTION LIST ---

    questions.append(normals[0])
    questions.append(normals[1])
    questions.append(SANITY_QUESTIONS[0])

    questions.append(normals[2])
    questions.append(SANITY_QUESTIONS[1])

    questions.append(normals[3])
    questions.append(SANITY_QUESTIONS[2])

    questions.append(normals[4])
    questions.append(SANITY_QUESTIONS[3])
    questions.append(SANITY_QUESTIONS[4])

    questions.append(make_introspection_question())

    # Normalize object names
    for q in questions:
        if isinstance(q, dict) and "object" in q:
            q["object"] = normalize_object_name(q["object"])


    return {
        "profile_id": "debug_profile",
        "questions": questions,
    }


def save_profile(profile, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
