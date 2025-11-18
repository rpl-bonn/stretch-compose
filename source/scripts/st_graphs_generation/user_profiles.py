# user_profiles.py
import random
from typing import Optional

BASE_PROFILES = {
    # ---- Single person ----
    "tidy_single": {
        "reloc_prob": 0.1,
        "misplace_prob": 0.05,
        "drawer_usage": 0.7
    },
    "messy_single": {
        "reloc_prob": 0.4,
        "misplace_prob": 0.5,
        "drawer_usage": 0.1
    },
    "habitual_single": {
        "reloc_prob": 0.2,
        "misplace_prob": 0.05,
        "drawer_usage": 0.5
    },
    "forgetful_single": {
        "reloc_prob": 0.3,
        "misplace_prob": 0.2,
        "drawer_usage": 0.4
    },
    "minimalist_single": {
        "reloc_prob": 0.1,
        "misplace_prob": 0.02,
        "drawer_usage": 0.6
    },

    # ---- Couples ----
    "couple_wfh": {
        "reloc_prob": 0.25,
        "misplace_prob": 0.1,
        "drawer_usage": 0.6
    },
    "couple_one_wfh": {
        "reloc_prob": 0.35,
        "misplace_prob": 0.2,
        "drawer_usage": 0.5
    },
    "couple_busy_week": {
        "reloc_prob": 0.4,
        "misplace_prob": 0.25,
        "drawer_usage": 0.4
    },
    "couple_mixed": {
        "reloc_prob": 0.3,
        "misplace_prob": 0.3,
        "drawer_usage": 0.3
    },
    "couple_messy": {
        "reloc_prob": 0.5,
        "misplace_prob": 0.35,
        "drawer_usage": 0.2
    }
}
def get_profile(name: str, random_seed: Optional[int] = None, variation: float = 0.05):
    """
    Return a profile with slight random variation.
    
    Args:
        name: profile key (e.g., "tidy_single").
        random_seed: optional seed for reproducibility.
        variation: max relative variation (e.g., 0.05 = ±5%).
    """
    if name not in BASE_PROFILES:
        raise ValueError(f"Unknown profile name: {name}")
    
    if random_seed is not None:
        random.seed(random_seed)
    
    base = BASE_PROFILES[name]
    profile = {}
    for key, val in base.items():
        delta = val * variation
        profile[key] = min(max(val + random.uniform(-delta, delta), 0.0), 1.0)
    return profile

def list_profiles():
    """Return list of all available profile names."""
    return list(BASE_PROFILES.keys())
