from pydantic import BaseModel, Field


class Reply(BaseModel):
    answer: str
    confidence: int = Field(ge=0, le=100)


def clean_json(text: str) -> str:
    """Remove ``` code fence lines that models often add around JSON."""
    lines = [
        line for line in text.strip().splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()