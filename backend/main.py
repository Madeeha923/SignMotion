import json
import hashlib
import math
import os
import pickle
import re
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- 1. SETUP ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
gemini_translator_available = bool(gemini_api_key and gemini_api_key != "your-gemini-api-key-here")

app = FastAPI(title="Signvrse Generative AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 150
MAX_RETRIEVAL_SEQUENCE_FRAMES = 180
CHUNK_COUNT = 40
CHUNK_EPSILON = 1e-5
MIN_FULL_RETRIEVAL_SCORE = 0.78
MIN_FULL_QUERY_COVERAGE = 0.75
MIN_FULL_CANDIDATE_COVERAGE = 0.65
MIN_SEGMENT_FRAMES = 10
RotationDict = Dict[str, float]
RVQ_TOKEN_KEYS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]

TOKEN_CORRECTIONS = {
    "ANYBODE": "ANYBODY",
    "ANYBODYE": "ANYBODY",
    "ANYONEE": "ANYONE",
    "FURAHI": "HAPPY",
    "KUZALIWA": "BIRTHDAY",
    "SIKU": "",
    "TOMMORROW": "TOMORROW",
    "TOMOROW": "TOMORROW",
    "2MORROW": "TOMORROW",
    "BIRTHDAYY": "BIRTHDAY",
    "HAPY": "HAPPY",
    "HELLOO": "HELLO",
}

FALLBACK_DROP_TOKENS = {
    "A",
    "AN",
    "AM",
    "ARE",
    "BE",
    "IS",
    "THE",
    "TO",
    "WAS",
    "WERE",
    "WILL",
}

FALLBACK_PHRASE_REPLACEMENTS = {
    "I AM": "ME",
    "I WILL": "",
    "I WANT TO": "ME WANT",
    "DO YOU": "YOU",
    "ARE YOU": "YOU",
}

CONTROL_SCALE_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    "spine_rotation": (0.10, 0.14, 0.08),
    "spine_upper_rotation": (0.12, 0.18, 0.10),
    "neck_rotation": (0.10, 0.18, 0.10),
    "head_rotation": (0.12, 0.20, 0.12),
    "right_shoulder_rotation": (0.18, 0.26, 0.16),
    "left_shoulder_rotation": (0.18, 0.26, 0.16),
    "right_arm_rotation": (0.38, 0.70, 0.45),
    "left_arm_rotation": (0.38, 0.70, 0.45),
    "right_forearm_rotation": (0.48, 0.38, 0.58),
    "left_forearm_rotation": (0.48, 0.38, 0.58),
    "right_hand_rotation": (0.28, 0.32, 0.42),
    "left_hand_rotation": (0.28, 0.32, 0.42),
}

FINGER_LAYOUT: Dict[str, float] = {
    "right_hand_curl": 0.55,
    "left_hand_curl": 0.55,
}

ARM_CONTROL_NAMES = {
    "right_shoulder_rotation",
    "left_shoulder_rotation",
    "right_arm_rotation",
    "left_arm_rotation",
    "right_forearm_rotation",
    "left_forearm_rotation",
    "right_hand_rotation",
    "left_hand_rotation",
}


def backend_path(*parts: str) -> str:
    return os.path.join(BACKEND_DIR, *parts)


def normalize_token(token: str) -> str:
    cleaned = token.upper().strip().replace("\u2019", "'").replace("\u00e2\u20ac\u2122", "'")
    cleaned = cleaned.strip(".,!?;:\"()[]{}")
    return TOKEN_CORRECTIONS.get(cleaned, cleaned)


def fallback_english_to_gloss(text: str) -> str:
    """Small offline translator used when the Gemini quota/API is unavailable."""
    normalized_text = " ".join(normalize_token(token) for token in text.split())

    for phrase, replacement in FALLBACK_PHRASE_REPLACEMENTS.items():
        normalized_text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, normalized_text)

    tokens = normalize_gloss_text(normalized_text)
    content_tokens = [token for token in tokens if token not in FALLBACK_DROP_TOKENS]
    return " ".join(content_tokens or tokens)


def legacy_lookup_vocab_index(word: str) -> int:
    normalized = normalize_token(word)
    candidates = [
        normalized,
        normalized.rstrip(".,!?"),
        normalized.replace("\u00e2\u20ac\u2122", "'"),
        normalized.rstrip(".,!?").replace("\u00e2\u20ac\u2122", "'"),
    ]
    for candidate in candidates:
        if candidate in vocab:
            return vocab[candidate]
    return 0


def normalize_gloss_text(text: str) -> List[str]:
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", text.upper().replace("//", " "))
    return [normalize_token(token) for token in cleaned.split() if normalize_token(token)]


def lookup_vocab_index(word: str) -> int:
    normalized = normalize_token(word)
    stripped = normalized.strip(".,!?;:\"()[]{}")
    candidates = [
        normalized,
        stripped,
        normalized.rstrip(".,!?"),
        stripped.rstrip("/"),
        f"{stripped}//",
        f"{stripped}?",
    ]
    for candidate in candidates:
        if candidate in vocab:
            return vocab[candidate]
    return 0


def gloss_match_score(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0

    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(query_set & candidate_set)
    if overlap == 0:
        return 0.0

    token_f1 = (2 * overlap) / (len(query_set) + len(candidate_set))
    sequence_ratio = SequenceMatcher(None, " ".join(query_tokens), " ".join(candidate_tokens)).ratio()
    return 0.75 * token_f1 + 0.25 * sequence_ratio


def gloss_match_coverage(query_tokens: List[str], candidate_tokens: List[str]) -> Tuple[float, float]:
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(query_set & candidate_set)
    query_coverage = overlap / len(query_set) if query_set else 0.0
    candidate_coverage = overlap / len(candidate_set) if candidate_set else 0.0
    return query_coverage, candidate_coverage


def find_best_retrieval_match(gloss: str) -> Tuple[Dict[str, Any] | None, float]:
    query_tokens = normalize_gloss_text(gloss)
    if not query_tokens:
        return None, 0.0

    best_entry = None
    best_score = 0.0

    for entry in retrieval_entries:
        score = gloss_match_score(query_tokens, entry["normalized_tokens"])
        if score > best_score:
            best_entry = entry
            best_score = score

    return best_entry, best_score
def build_retrieval_latent_frames(entry: Dict[str, Any]) -> np.ndarray:
    if "latent_frames" in entry:
        return entry["latent_frames"]

    item = entry["item"]
    frame_count = len(item["tokens"][RVQ_TOKEN_KEYS[0]])
    latent_frames = np.zeros((frame_count, 256), dtype=np.float32)

    for frame_index in range(frame_count):
        for quantizer_index, token_key in enumerate(RVQ_TOKEN_KEYS):
            token_id = item["tokens"][token_key][frame_index]
            latent_frames[frame_index] += rvq_codebooks[quantizer_index][:, token_id]

    entry["latent_frames"] = latent_frames
    return latent_frames


def retrieval_match_is_strong(gloss: str, entry: Dict[str, Any] | None, score: float) -> bool:
    if entry is None or not rvq_codebooks:
        return False

    query_coverage, candidate_coverage = gloss_match_coverage(
        normalize_gloss_text(gloss),
        entry["normalized_tokens"],
    )
    return (
        score >= MIN_FULL_RETRIEVAL_SCORE
        and query_coverage >= MIN_FULL_QUERY_COVERAGE
        and candidate_coverage >= MIN_FULL_CANDIDATE_COVERAGE
    )


def find_best_segment_match(query_tokens: List[str], target_index: int) -> Tuple[Dict[str, Any] | None, int, float]:
    target_token = query_tokens[target_index]
    query_set = set(query_tokens)
    best_entry = None
    best_token_index = 0
    best_score = 0.0

    for entry in retrieval_entries:
        candidate_tokens = entry["normalized_tokens"]
        if target_token not in candidate_tokens:
            continue

        overlap = len(query_set & set(candidate_tokens))
        context_score = overlap / len(query_set)
        compactness_score = 1.0 / max(len(candidate_tokens), 1)
        sequence_score = SequenceMatcher(
            None,
            " ".join(query_tokens),
            " ".join(candidate_tokens),
        ).ratio()
        score = 0.6 * context_score + 0.25 * compactness_score + 0.15 * sequence_score

        if score > best_score:
            best_entry = entry
            best_token_index = candidate_tokens.index(target_token)
            best_score = score

    return best_entry, best_token_index, best_score


def slice_retrieval_token_frames(entry: Dict[str, Any], token_index: int) -> np.ndarray:
    latent_frames = build_retrieval_latent_frames(entry)
    token_count = max(len(entry["normalized_tokens"]), 1)
    frame_count = latent_frames.shape[0]

    start = int((token_index / token_count) * frame_count)
    end = int(((token_index + 1) / token_count) * frame_count)
    center = (start + end) // 2
    half_width = max((end - start) // 2, MIN_SEGMENT_FRAMES // 2)

    start = max(0, center - half_width)
    end = min(frame_count, center + half_width)
    if end <= start:
        end = min(frame_count, start + MIN_SEGMENT_FRAMES)

    return latent_frames[start:end]


def build_segmented_retrieval_sequence(gloss: str) -> Tuple[np.ndarray | None, str]:
    if not rvq_codebooks:
        return None, "unavailable"

    query_tokens = normalize_gloss_text(gloss)
    if not query_tokens:
        return None, "empty"

    segments = []
    sources = []
    for token_index, token in enumerate(query_tokens):
        entry, entry_token_index, score = find_best_segment_match(query_tokens, token_index)
        if entry is None:
            continue

        segments.append(slice_retrieval_token_frames(entry, entry_token_index))
        sources.append(f"{token}->{entry['gloss']}:{score:.2f}")

    if len(segments) < max(1, int(len(query_tokens) * 0.5)):
        return None, "too-few-segments"

    source_frames = np.concatenate(segments, axis=0)
    if source_frames.shape[0] > MAX_RETRIEVAL_SEQUENCE_FRAMES:
        sampled_indices = np.linspace(
            0,
            source_frames.shape[0] - 1,
            MAX_RETRIEVAL_SEQUENCE_FRAMES,
        ).astype(int)
        source_frames = source_frames[sampled_indices]

    return source_frames, "segments:" + " | ".join(sources[:6])


def summarize_motion_frames(motion_frames: np.ndarray) -> np.ndarray:
    chunks = np.array_split(motion_frames, CHUNK_COUNT, axis=1)
    chunk_signals = np.stack([chunk.mean(axis=1) for chunk in chunks], axis=1)
    centered = chunk_signals - chunk_signals.mean(axis=0, keepdims=True)
    normalized = centered / (centered.std(axis=0, keepdims=True) + CHUNK_EPSILON)
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    velocity = velocity / (velocity.std(axis=0, keepdims=True) + CHUNK_EPSILON)

    # Keep the motion centered around the avatar rest pose so sentence changes
    # show up as different movement directions instead of one shared bias.
    return (
        0.7 * np.tanh(normalized * 1.2) +
        0.3 * np.tanh(velocity * 1.0)
    )


def build_rotation_frame(
    signal_frame: np.ndarray,
    motion_style: Dict[str, Any] | None = None,
    frame_index: int = 0,
    frame_count: int = 1,
) -> Dict[str, RotationDict | float]:
    frame_data: Dict[str, RotationDict | float] = {}
    signal = signal_frame.astype(np.float32, copy=False)
    progress = frame_index / max(frame_count - 1, 1)
    signal_energy = float(np.tanh(np.linalg.norm(signal) / math.sqrt(CHUNK_COUNT)))

    for control_name, (scale_x, scale_y, scale_z) in CONTROL_SCALE_LAYOUT.items():
        projected = np.tanh(control_projections[control_name] @ signal)
        if motion_style:
            projected = projected * motion_style["control_signs"][control_name]
            projected = projected * motion_style["control_gain"][control_name]

            sweep_amount = motion_style["control_sweep"][control_name]
            if sweep_amount:
                phase = motion_style["control_phase"][control_name]
                sweep = sweep_amount * signal_energy * np.array(
                    [
                        math.sin(math.tau * progress + phase),
                        math.sin(math.tau * progress * 0.65 + phase + 1.2),
                        math.cos(math.tau * progress + phase),
                    ],
                    dtype=np.float32,
                )
                projected = projected + sweep

        frame_data[control_name] = {
            "x": float(np.clip(projected[0] * scale_x, -1.2, 1.2)),
            "y": float(np.clip(projected[1] * scale_y, -1.2, 1.2)),
            "z": float(np.clip(projected[2] * scale_z, -1.2, 1.2)),
        }

    for control_name, scale in FINGER_LAYOUT.items():
        projected = np.tanh(finger_projections[control_name] @ signal)
        if motion_style:
            projected = projected * motion_style["finger_signs"][control_name]
        frame_data[control_name] = float(np.clip(projected * scale, -0.9, 0.9))

    return frame_data


def make_projection_bank(output_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(output_size, CHUNK_COUNT)).astype(np.float32)
    projection /= np.linalg.norm(projection, axis=1, keepdims=True) + CHUNK_EPSILON
    return projection


def stable_text_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def build_motion_style(gloss: str) -> Dict[str, Any]:
    seed = stable_text_seed(" ".join(normalize_gloss_text(gloss)))
    rng = np.random.default_rng(seed)
    dominant_side = "right" if seed % 2 == 0 else "left"

    control_signs = {}
    control_gain = {}
    control_sweep = {}
    control_phase = {}

    for control_name in CONTROL_SCALE_LAYOUT:
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=3)
        side_gain = 1.0
        if control_name.startswith(dominant_side):
            side_gain = 1.18
        elif control_name.startswith("left") or control_name.startswith("right"):
            side_gain = 0.82

        control_signs[control_name] = signs
        control_gain[control_name] = float(rng.uniform(0.85, 1.2) * side_gain)
        control_sweep[control_name] = float(rng.uniform(0.06, 0.18) if control_name in ARM_CONTROL_NAMES else 0.0)
        control_phase[control_name] = float(rng.uniform(0.0, math.tau))

    finger_signs = {
        control_name: float(rng.choice(np.array([-1.0, 1.0], dtype=np.float32)))
        for control_name in FINGER_LAYOUT
    }

    return {
        "control_signs": control_signs,
        "control_gain": control_gain,
        "control_sweep": control_sweep,
        "control_phase": control_phase,
        "finger_signs": finger_signs,
        "dominant_side": dominant_side,
    }


control_projections = {
    control_name: make_projection_bank(3, 100 + index)
    for index, control_name in enumerate(CONTROL_SCALE_LAYOUT)
}
finger_projections = {
    control_name: make_projection_bank(1, 400 + index)[0]
    for index, control_name in enumerate(FINGER_LAYOUT)
}

# --- 2. GENERATIVE AI CLASSES ---
# We must define the model architecture so PyTorch knows how to load the .pth files
class MotionAutoencoder(nn.Module):
    def __init__(self, input_dim=668, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class TextToMotionBrain(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, latent_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.latent_projector = nn.Linear(512, MAX_FRAMES * latent_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        context = torch.cat((hidden[-2], hidden[-1]), dim=1) 
        latent_sequence = self.latent_projector(context)
        return latent_sequence.view(-1, MAX_FRAMES, 512)

# --- 3. LOAD MODELS INTO MEMORY ---
print("Loading KSL Vocabulary...")
try:
    with open(backend_path("ksl_vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
except FileNotFoundError:
    print("WARNING: ksl_vocab.json not found! Please ensure it's in the same directory.")
    vocab = {"<PAD>": 0}

print("Booting up KSL Generative Engine...")
autoencoder = MotionAutoencoder(input_dim=668, latent_dim=512).to(DEVICE)
autoencoder.load_state_dict(
    torch.load(backend_path("ksl_movement_dictionary.pth"), map_location=DEVICE, weights_only=True)
)
autoencoder.eval()

brain = TextToMotionBrain(vocab_size=len(vocab)).to(DEVICE)
brain.load_state_dict(
    torch.load(backend_path("ksl_text_brain.pth"), map_location=DEVICE, weights_only=True)
)
brain.eval()

print("Loading retrieval gloss metadata...")
try:
    with open(backend_path("data", "index_meta.pkl"), "rb") as f:
        retrieval_meta = pickle.load(f)

    rvq_checkpoint = torch.load(
        backend_path("ml", "rvq_vae_best.pth"),
        map_location="cpu",
        weights_only=True,
    )
    rvq_codebooks = [
        rvq_checkpoint["model_state_dict"][f"rvq.quantizers.{index}.embedding"].cpu().numpy()
        for index in range(len(RVQ_TOKEN_KEYS))
    ]
    retrieval_entries = [
        {
            "item": item,
            "gloss": item["gloss"],
            "normalized_tokens": normalize_gloss_text(item["gloss"]),
        }
        for item in retrieval_meta.values()
    ]
except FileNotFoundError:
    print("WARNING: retrieval metadata not found. Falling back to text-only generation.")
    retrieval_entries = []
    rvq_codebooks = []


# --- 4. LANGGRAPH STATE & AGENTS ---
class GraphState(TypedDict):
    english_text: str
    ksl_gloss: str
    final_3d_data: List[Dict]

def translator_agent(state: GraphState):
    """Node 1: Translates English to authentic KSL Gloss using Gemini."""
    global gemini_translator_available
    text = state["english_text"]

    if not gemini_translator_available or client is None:
        real_gloss = fallback_english_to_gloss(text)
        print(f"[Translator Agent] offline fallback '{text}' -> '{real_gloss}'")
        return {"ksl_gloss": real_gloss}
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate to KSL Gloss:\n{text}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert linguist in Kenyan Sign Language (KSL). "
                    "Output only KSL gloss tokens written as English words in ALL CAPS. "
                    "Do not translate into Swahili, Kiswahili, or any spoken-language sentence. "
                    "Do not output words like FURAHI, SIKU, KUZALIWA, ASANTE, or JAMBO. "
                    "Use short gloss order, drop 'to be' verbs and articles, and use no punctuation. "
                    "Examples: happy birthday -> HAPPY BIRTHDAY; I am going to college -> ME GO COLLEGE; "
                    "happy birthday to you -> HAPPY BIRTHDAY YOU."
                ),
                temperature=0.1,
            ),
        )
        real_gloss = (response.text or "").strip()
        if not real_gloss:
            raise ValueError("Gemini returned an empty gloss")
        real_gloss = " ".join(normalize_gloss_text(real_gloss))
    except Exception as e:
        print(f"Gemini Error: {e}")
        error_text = str(e).lower()
        if "api_key_invalid" in error_text or "api key not valid" in error_text or "quota" in error_text or "429" in error_text:
            gemini_translator_available = False
        real_gloss = fallback_english_to_gloss(text)
    
    print(f"[Translator Agent] '{text}' -> '{real_gloss}'")
    return {"ksl_gloss": real_gloss}

def generator_agent(state: GraphState):
    """Node 2: Generates 3D Coordinates using our custom trained AI models."""
    gloss = state["ksl_gloss"]
    
    # 1. Convert Text to Tensor
    words = gloss.split()
    text_indices = [lookup_vocab_index(word) for word in words if word.strip()]
    if not text_indices:
        text_indices = [0]
    text_tensor = torch.tensor([text_indices], dtype=torch.long).to(DEVICE)

    # 2. Prefer retrieval over generic synthesis when we can match the gloss.
    matched_entry, matched_score = find_best_retrieval_match(gloss)
    motion_source = "text-brain"

    if retrieval_match_is_strong(gloss, matched_entry, matched_score):
        source_frames = build_retrieval_latent_frames(matched_entry)
        motion_source = f"retrieval:{matched_entry['gloss']}"
    else:
        segmented_frames, segment_source = build_segmented_retrieval_sequence(gloss)
        if segmented_frames is not None:
            source_frames = segmented_frames
            motion_source = segment_source
        else:
            with torch.no_grad():
                latent_vector = brain(text_tensor)
                source_frames = autoencoder.decoder(latent_vector).squeeze(0).cpu().numpy()
            motion_source = f"text-brain:{segment_source}"

    control_frames = summarize_motion_frames(source_frames)
    motion_style = build_motion_style(gloss)

    # 3. Format for React Frontend
    formatted_frames = [
        build_rotation_frame(frame, motion_style, frame_index, len(control_frames))
        for frame_index, frame in enumerate(control_frames)
    ]

    print(
        f"[Generator Agent] Rendered {len(formatted_frames)} frames "
        f"using {motion_source} (score={matched_score:.3f})."
    )
    return {"final_3d_data": formatted_frames}

# --- 5. COMPILE GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("translator", translator_agent)
workflow.add_node("generator", generator_agent)

workflow.set_entry_point("translator")
workflow.add_edge("translator", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()

# --- 6. API ENDPOINTS ---
class TranslateRequest(BaseModel):
    sentence: str

@app.post("/translate-to-sign")
async def generate_sign_language(req: TranslateRequest):
    """The frontend UI calls this endpoint when a user clicks 'Translate'"""
    initial_state = {"english_text": req.sentence}
    final_state = app_graph.invoke(initial_state)
    
    return {
        "status": "success",
        "gloss_used": final_state["ksl_gloss"],
        "final_3d_data": final_state["final_3d_data"]
    }

if __name__ == "__main__":
    print("Starting Signvrse Generative API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

