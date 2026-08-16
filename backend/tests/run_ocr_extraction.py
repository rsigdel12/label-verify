import pathlib
import time
import asyncio

import json

from app.comparison.matcher import compare_fields
from app.extraction.vision_client import extract_label_fields

BASE = pathlib.Path(__file__).resolve().parent / "fixtures"


async def run_one(path):
    data = path.read_bytes()
    start = time.perf_counter()
    try:
        result = await extract_label_fields(data)
        elapsed = time.perf_counter() - start
        return {
            "file": path.name,
            "elapsed": round(elapsed, 3),
            "result": result.model_dump(),
        }
    except Exception as e:
        return {"file": path.name, "error": str(e)}


async def main():
    pngs = sorted(BASE.glob("fixture_0*.png"))
    failures = []
    for p in pngs:
        result = await run_one(p)
        if "error" in result:
            failures.append(f"{p.name}: {result['error']}")
            print(f"ERROR  {p.name:<38} {result['error']}")
            continue
        metadata = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        extracted = result["result"]
        from app.extraction.schema import ExtractedLabel

        comparison = compare_fields(
            ExtractedLabel(**extracted), metadata["application_data"]
        )
        statuses = {field: item["status"] for field, item in comparison.items()}
        expected = metadata["expected_statuses"]
        outcome = "PASS" if statuses == expected else "FAIL"
        print(f"{outcome:<5}  {p.name:<38} {result['elapsed']:.3f}s")
        if result["elapsed"] >= 5:
            failures.append(
                f"{p.name}: exceeded five-second target ({result['elapsed']:.3f}s)"
            )
        if statuses != expected:
            failures.append(f"{p.name}: expected {expected}, got {statuses}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
