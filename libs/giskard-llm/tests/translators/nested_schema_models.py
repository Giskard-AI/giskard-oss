"""Shared nested Pydantic models for structured-output translator tests."""

from pydantic import BaseModel


class NestedInnerModel(BaseModel):
    value: str


class NestedOutputModel(BaseModel):
    inner: NestedInnerModel


class FlatOutputModel(BaseModel):
    value: int
    count: int
