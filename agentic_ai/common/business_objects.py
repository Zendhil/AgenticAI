# Standard library imports
# Third party imports
# App imports

from pydantic import BaseModel, Field
from typing import Literal, Optional

class Salary(BaseModel):
    salary: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, pattern=r"^[A-Z]{3}$")
    period: Optional[Literal["hour", "month", "year"]] = None

class JobPosting(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    remote_policy: Literal["remote", "hybrid", "onsite", "unknown"]
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unknown"]
    salary: Optional[Salary] = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[int] = Field(None, ge=0, le=40)