"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Core PromptForge schemas

class Project(BaseModel):
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Short description")
    owner_id: Optional[str] = Field(None, description="User id of the owner")
    tags: List[str] = Field(default_factory=list, description="Project tags")

class Prompt(BaseModel):
    project_id: str = Field(..., description="Associated project id")
    title: str = Field(..., description="Prompt title")
    instructions: str = Field(..., description="User instructions / goal")
    context: Optional[str] = Field(None, description="Additional background context")
    audience: Optional[str] = Field(None, description="Target audience or role")
    constraints: Optional[str] = Field(None, description="Constraints / rules")
    format: Optional[str] = Field(None, description="Desired output format")
    examples: Optional[str] = Field(None, description="Few-shot examples if any")
    optimized_prompt: Optional[str] = Field(None, description="Generated optimized prompt text")
    score: Optional[float] = Field(None, ge=0, le=1, description="Heuristic quality score 0-1")
    model: Optional[str] = Field(None, description="Recommended model for this prompt")
    version: int = Field(1, description="Version number for this prompt")

class Run(BaseModel):
    prompt_id: str = Field(..., description="Prompt id this run is for")
    model: str = Field(..., description="Model used during the run")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(512, ge=1, le=8192)
    input_prompt: str = Field(..., description="Prompt text sent to model")
    output: Optional[str] = Field(None, description="Model output text")
    latency_ms: Optional[int] = Field(None, description="Latency in milliseconds")
    cost_usd: Optional[float] = Field(None, description="Approximate cost in USD")
    score: Optional[float] = Field(None, ge=0, le=1, description="Heuristic or LLM score 0-1")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Extra run metadata")

class Template(BaseModel):
    key: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Human-readable name")
    content: str = Field(..., description="Jinja-like template string")
    description: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)

# Example legacy schemas (kept for reference)
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = None
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str
    in_stock: bool = True
