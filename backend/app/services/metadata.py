"""Parsing + validation of the cow registration data (breed, DOB, dam, sire, milk yields).

Accepted DOB formats: 21/08/2026, 21/08/26, 21-08-2026, 2026-08-21.
Two-digit years are read as 20xx (a DOB can never be in the future).
"""
import re
from datetime import date, datetime
from typing import Optional

BREEDS = ["Gir", "Sahiwal", "Kankrej", "Red Sindhi", "Tharparkar", "Rathi", "Holstein Friesian", "Jersey",
          "HF Cross", "Jersey Cross", "Murrah (buffalo)", "Other"]

_FIELDS_TEXT = {"breed": 60, "dam_id": 40, "sire_id": 120, "name": 120, "owner": 120}


class MetadataError(ValueError):
    """Raised with a message that is safe to show to the user."""


def parse_dob(value: Optional[str]) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            d = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        if d > date.today():
            raise MetadataError(f"Date of birth {d:%d/%m/%Y} is in the future.")
        if d.year < 1990:
            raise MetadataError(f"Date of birth {d:%d/%m/%Y} looks wrong (use dd/mm/yyyy).")
        return d
    raise MetadataError(f"Date of birth '{value}' is not valid. Use dd/mm/yyyy, e.g. 21/08/2026.")


def parse_milk(value: Optional[str], label: str) -> Optional[float]:
    """Milk yield in kg. Accepts '3573.83', '3,573.83', '3573.83 kg'."""
    s = (value or "").strip().lower().replace("kgs", "").replace("kg", "").replace(",", "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        raise MetadataError(f"{label} '{value}' is not a number.")
    if v != v or v < 0 or v > 30000:
        raise MetadataError(f"{label} {v} kg is outside the realistic range (0 - 30000 kg).")
    return round(v, 2)


def clean_text(value: Optional[str], field: str) -> str:
    s = re.sub(r"\s+", " ", (value or "").strip())
    if len(s) > _FIELDS_TEXT[field]:
        raise MetadataError(f"{field} is too long (max {_FIELDS_TEXT[field]} characters).")
    return s


def clean_dam_id(value: Optional[str]) -> str:
    """Dam ID is kept as TEXT (leading zeros matter). Spaces / dashes are removed from numeric tags."""
    s = clean_text(value, "dam_id")
    compact = re.sub(r"[\s\-]", "", s)
    return compact if compact.isdigit() else s


def clean_sire_id(value: Optional[str]) -> str:
    return clean_text(value, "sire_id").upper()      # e.g. 'NDD ESPECTACULO'


def build_metadata(breed, dob, dam_id, dam_milk_305, sire_id, sire_dam_milk) -> dict:
    """Validate all registration fields and return a dict ready for the database."""
    return dict(
        breed=clean_text(breed, "breed"),
        dob=parse_dob(dob),
        dam_id=clean_dam_id(dam_id),
        dam_milk_305=parse_milk(dam_milk_305, "Dam's 305-day milk yield"),
        sire_id=clean_sire_id(sire_id),
        sire_dam_milk=parse_milk(sire_dam_milk, "Sire's dam milk yield"),
    )


def age_text(dob: Optional[date]) -> str:
    if not dob:
        return ""
    days = (date.today() - dob).days
    months = days // 30
    if months < 1:
        return f"{days} day(s)"
    if months < 24:
        return f"{months} month(s)"
    return f"{months // 12} yr {months % 12} mo"
