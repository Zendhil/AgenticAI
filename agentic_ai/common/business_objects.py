# Standard library imports
# Third party imports
# App imports

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any, Dict

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


# ===== RAG Data Models =====

class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    source: Optional[str] = Field(None, description="Source filename")
    file_path: Optional[str] = Field(None, description="Full file path")
    file_type: Optional[str] = Field(None, description="File type extension")
    page: Optional[str] = Field(None, description="Page number or label")
    section: Optional[str] = Field(None, description="Section name")
    doc_id: Optional[str] = Field(None, description="Document ID")
    index: Optional[int] = Field(None, description="Chunk index in document")

class Chunk(BaseModel):
    """Represents a text chunk extracted from a document."""
    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="The actual text content")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata, description="Chunk metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "text": "Apple Inc. was founded in 1976...",
                "metadata": {
                    "source": "company_history.pdf",
                    "page": "1",
                    "index": 0
                }
            }
        }

class Relation(BaseModel):
    """Represents a relationship between two entities."""
    type: str = Field(..., description="Relationship type (e.g., CEO_OF, FOUNDED)")
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship properties")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "CEO_OF",
                "source": "Steve Jobs",
                "target": "Apple Inc.",
                "properties": {}
            }
        }

class Entity(BaseModel):
    """Represents a named entity."""
    name: str = Field(..., description="Entity name")
    label: str = Field(..., description="Entity type/label (e.g., Person, Company)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional entity properties")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Steve Jobs",
                "label": "Person",
                "properties": {"source": "company_history.pdf"}
            }
        }

class EntityContext(BaseModel):
    """Complete context for an entity including relationships and source chunks."""
    entity: Entity = Field(..., description="The entity information")
    relationships: List[Relation] = Field(default_factory=list, description="Outgoing relationships from this entity")
    chunks: List[Chunk] = Field(default_factory=list, description="Text chunks where this entity was mentioned")

    class Config:
        json_schema_extra = {
            "example": {
                "entity": {
                    "name": "Steve Jobs",
                    "label": "Person",
                    "properties": {}
                },
                "relationships": [
                    {"type": "CEO_OF", "source": "Steve Jobs", "target": "Apple Inc.", "properties": {}}
                ],
                "chunks": [
                    {
                        "id": "chunk_1",
                        "text": "Steve Jobs was the CEO...",
                        "metadata": {"source": "bio.pdf"}
                    }
                ]
            }
        }