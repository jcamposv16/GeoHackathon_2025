# ============================================================
# src/well_mapping.py
# Bidirectional well name mapping
# Maps folder names to well IDs and normalizes queries
# ============================================================

import re

# Complete well mapping for SPE GeoHackathon 2025
WELL_MAPPING = {
    # Folder name → list of well IDs in that folder
    "Well 1": ["ADK-GT-01", "ADK-GT-01-S1"],
    "Well 2": ["HAG-GT-01", "HAG-GT-02"],
    "Well 3": ["MDM-GT-06", "MDM-GT-06-S1", "MDM-GT-06-S2"],
    "Well 4": ["NLW-GT-02-S1"],
    "Well 5": ["NLW-GT-03", "NLW-GT-03-S1"],
    "Well 6": ["LIR-GT-01"],
    "Well 7": ["BRI-GT-01"],
    "Well 8": ["MSD-GT-01"],
}

# Well type information
WELL_INFO = {
    "ADK-GT-01":    {"type": "geothermal", "sidetrack": False},
    "ADK-GT-01-S1": {"type": "geothermal", "sidetrack": True},
    "HAG-GT-01":    {"type": "producer",   "sidetrack": False},
    "HAG-GT-02":    {"type": "injector",   "sidetrack": False},
    "MDM-GT-06":    {"type": "geothermal", "sidetrack": False},
    "MDM-GT-06-S1": {"type": "geothermal", "sidetrack": True},
    "MDM-GT-06-S2": {"type": "geothermal", "sidetrack": True,
                     "note": "last completion"},
    "NLW-GT-02-S1": {"type": "geothermal", "sidetrack": True},
    "NLW-GT-03":    {"type": "geothermal", "sidetrack": False},
    "NLW-GT-03-S1": {"type": "geothermal", "sidetrack": True},
    "LIR-GT-01":    {"type": "geothermal", "sidetrack": False},
    "BRI-GT-01":    {"type": "geothermal", "sidetrack": False},
    "MSD-GT-01":    {"type": "geothermal", "sidetrack": False},
}

# Alternative names found in some reports → canonical well ID
WELL_ALIASES = {
    "DEN HAAG GT-01":  "HAG-GT-01",
    "DEN HAAG GT-02":  "HAG-GT-02",
    "DEN HAAG-GT-01":  "HAG-GT-01",
    "DEN HAAG-GT-02":  "HAG-GT-02",
    "HAAG-GT-01":      "HAG-GT-01",
    "HAAG-GT-02":      "HAG-GT-02",
    "MSD-GT-01-P":     "MSD-GT-01",
    "MAASDIJK-GT-01":  "MSD-GT-01",
}

# Reverse map: well ID → folder name
ID_TO_FOLDER = {}
for folder, ids in WELL_MAPPING.items():
    for well_id in ids:
        ID_TO_FOLDER[well_id] = folder

# All known well IDs as a flat list
ALL_WELL_IDS = list(ID_TO_FOLDER.keys())


def normalize_query(query: str) -> str:
    """
    Replace folder references like 'Well 1' with all corresponding well IDs.
    Sorted longest-first so 'Well 10' cannot match before 'Well 1'.
    Also resolves report aliases (e.g. 'DEN HAAG GT-01' → 'HAG-GT-01').
    Example: 'Well 5 surface location' →
             'NLW-GT-03 NLW-GT-03-S1 surface location'
    """
    normalized = query
    for folder in sorted(WELL_MAPPING.keys(), key=len, reverse=True):
        ids = WELL_MAPPING[folder]
        replacement = " ".join(ids)
        pattern = re.compile(re.escape(folder), re.IGNORECASE)
        normalized = pattern.sub(replacement, normalized)
    # Resolve aliases after folder expansion (longest alias first)
    for alias in sorted(WELL_ALIASES.keys(), key=len, reverse=True):
        canonical = WELL_ALIASES[alias]
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        normalized = pattern.sub(canonical, normalized)
    return normalized


def get_well_ids_from_query(query: str) -> list:
    """
    Extract all well IDs mentioned in a query, including via folder references
    and report aliases.  Case-insensitive.
    """
    found = []
    q_upper = query.upper()

    for well_id in ALL_WELL_IDS:
        if well_id.upper() in q_upper:
            found.append(well_id)

    for folder, ids in WELL_MAPPING.items():
        pattern = re.compile(re.escape(folder), re.IGNORECASE)
        if pattern.search(query):
            for wid in ids:
                if wid not in found:
                    found.append(wid)

    # Resolve aliases so e.g. "DEN HAAG GT-01" yields "HAG-GT-01"
    for alias, canonical in WELL_ALIASES.items():
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        if pattern.search(query) and canonical not in found:
            found.append(canonical)

    return found


def get_folder_for_well(well_id: str) -> str:
    """Return the folder name for a given well ID."""
    return ID_TO_FOLDER.get(well_id, "Unknown")


def get_all_ids_in_same_folder(well_id: str) -> list:
    """
    Return all well IDs that share the same folder.
    Useful for HAG-GT-01/HAG-GT-02 which are in Well 2.
    """
    folder = get_folder_for_well(well_id)
    return WELL_MAPPING.get(folder, [well_id])


if __name__ == "__main__":
    print("Well mapping loaded:")
    for folder, ids in WELL_MAPPING.items():
        print(f"  {folder} → {ids}")

    tests = [
        "What is the surface location of Well 1?",
        "What is the surface location of Well 5?",
        "What is the surface location of well 8?",
        "Summarise Well 3 completion",
        "What is the casing scheme for HAG-GT-02?",
    ]
    print("\nNormalize tests:")
    for t in tests:
        print(f"  IN:  {t}")
        print(f"  OUT: {normalize_query(t)}")
        print(f"  IDs: {get_well_ids_from_query(t)}")
        print()
