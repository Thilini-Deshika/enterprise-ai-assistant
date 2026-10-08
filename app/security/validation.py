"""Input validation: API request models and generic string hygiene."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


def clean_text(text: str) -> str:
    return _CONTROL.sub("", text).strip()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str

    @field_validator("message")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = clean_text(v)
        if not v:
            raise ValueError("message must not be empty")
        return v

    @field_validator("session_id")
    @classmethod
    def _sid(cls, v: str) -> str:
        if not SESSION_ID_RE.match(v):
            raise ValueError("session_id must be 8-64 chars of [A-Za-z0-9_-]")
        return v


class ApprovalRequest(BaseModel):
    session_id: str
    approve: bool
    comment: str = Field(default="", max_length=500)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str = Field(max_length=64)
    rating: int = Field(ge=-1, le=1)  # -1 thumbs down, 1 thumbs up
    comment: str = Field(default="", max_length=1000)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
