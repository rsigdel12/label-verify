from pydantic import BaseModel, ConfigDict


class ExtractedLabel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    brand_name: str | None = None
    class_type: str | None = None
    alcohol_content: str | None = None
    net_contents: str | None = None
    warning_statement: str | None = None
