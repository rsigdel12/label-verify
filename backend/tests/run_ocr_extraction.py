import json
import pathlib
import time
import asyncio

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
    results = []
    for p in pngs:
        results.append(await run_one(p))
    out = BASE / "ocr_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
