from pydantic import BaseModel, Field


class ApplicationSubmission(BaseModel):
    brand_name: str | None = Field(default=None)
    class_type: str | None = Field(default=None)
    alcohol_content: str | None = Field(default=None)
    net_contents: str | None = Field(default=None)
    warning_statement: str | None = Field(default=None)
