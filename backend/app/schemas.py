from typing import Optional

from pydantic import BaseModel


class AnimalUpdate(BaseModel):
    """All fields optional - only the ones that are sent are changed. Dates as dd/mm/yyyy."""
    name: Optional[str] = None
    owner: Optional[str] = None
    breed: Optional[str] = None
    dob: Optional[str] = None
    dam_id: Optional[str] = None
    dam_milk_305: Optional[str] = None
    sire_id: Optional[str] = None
    sire_dam_milk: Optional[str] = None
