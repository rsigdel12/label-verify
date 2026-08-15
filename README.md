# Label Verify

A standalone proof of concept for comparing alcohol label artwork with an approved application. Reviewers upload one or more label images, enter the application values, and receive field-level `pass`, `fail`, or `needs_review` results with measured processing time.

Deployed application: <https://label-verify-p15v.onrender.com/>

## What the prototype covers

| Stakeholder need | Implementation |
| --- | --- |
| Obvious workflow for mixed technical comfort levels | One two-step screen, drag-and-drop upload, plain-language statuses, keyboard labels, responsive layout, and a one-click working sample |
| Results in about five seconds | Bundled ONNX OCR avoids a network round trip; the clean fixture takes about 2.6 seconds initially and 1.6–1.8 seconds warm locally |
| Compare label artwork with an application | The application form is the expected record; OCR text from the image is the actual value |
| Required sample fields | Brand name, class/type, alcohol content, net contents, and complete government warning |
| Human judgment for near matches | Brand casing/spacing is tolerant; a close but nonidentical regulated class/type is `needs_review` rather than silently approved |
| Exact warning language and capitalization | Warning comparison preserves case, punctuation, spelling, numbering, and order; visual/OCR whitespace is ignored |
| Imperfect photos | Test coverage includes 7-degree skew and simulated glare |
| Peak-season batch work | Multi-select in the same uploader calls the batch API and shows results per file |
| Restricted outbound network | The default extraction path is local RapidOCR + ONNX Runtime; no external API or secret is required |
| Prototype privacy | Uploads are validated, processed in memory, and not retained |
| Useful error handling | Unsupported, empty, oversized, corrupt, unavailable, timeout, and per-file batch failures return actionable messages |

The app intentionally assists rather than replaces the compliance agent. The warning's wording and capitalization are checked, but OCR text alone cannot prove that the `GOVERNMENT WARNING:` heading is visually bold. The reviewer must confirm styling and any requirements outside the five prototype fields.

### Fixture validity

The generated distilled-spirits fixtures follow the current TTB examples for the fields in scope: brand, class/type, and alcohol content appear together; net contents uses the accepted `750 mL` form; and the standard warning heading is uppercase and bold while the warning body is regular weight. The domestic label also includes `DISTILLED AND BOTTLED BY ACME SPIRITS — LOUISVILLE, KENTUCKY`; country of origin is not applicable to this domestic example. See TTB's [mandatory distilled-spirits information](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-brand-label), [health warning](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-health-warning), and [net contents](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-net-contents) guidance.

## Architecture and decisions

- `backend/app/` — FastAPI routes, validation, local/provider extraction, and comparison rules.
- `frontend/` — dependency-free HTML, CSS, and JavaScript served by FastAPI.
- `backend/tests/fixtures/` — eight generated label images with JSON application data and intended results.
- `frontend/assets/sample-label.png` — self-contained sample used by the live demonstration.

Images are normalized with Pillow, limited to 20 megapixels, and resized to a 900-pixel bound before OCR. RapidOCR's detector is capped at 640 pixels and ONNX Runtime uses one inference thread to control peak memory on small deployment instances. OCR runs in a worker thread so inference does not block the async server. A shared, lazily initialized engine avoids reloading the models on every request. Access to the engine is locked because a single inference session is reused safely. Batch requests preserve input order and report failures per file.

An optional OpenAI-compatible vision provider and a system Tesseract installation remain as fallback paths. The local engine runs first to keep the normal path fast and functional on Render without secrets or outbound access.

### Comparison rules

- Brand name: case-insensitive fuzzy comparison with a 90% threshold.
- Class/type: exact normalized text passes; close fuzzy text requires review; materially different text fails.
- Alcohol content: numeric ABV is compared, so equivalent formats such as `45% Alc. by Vol.` and `45% Alcohol by Volume` pass.
- Net contents: metric units are normalized, so `750 mL` and `0.75 L` are equivalent.
- Government warning: case, punctuation, spelling, numbering, and word order must match; whitespace caused by wrapping or OCR segmentation is ignored.
- Missing application or extraction values require manual review.

## Run locally

Python 3.11 or newer is required.

```powershell
cd label-verify\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. FastAPI serves both the API and frontend, so a separate static-file server is neither needed nor recommended.

No environment variable is required. To enable the optional provider fallback, set:

- `LLM_API_KEY`
- `LLM_BASE_URL` (defaults to `https://api.openai.com/v1`)
- `LLM_MODEL` (defaults to `gpt-4o-mini`)

## Test and evaluate

Run the automated suite:

```powershell
cd label-verify\backend
python -m pytest -q
```

The suite contains 30 tests, including real local OCR through both single and batch multipart endpoints. It also covers matching semantics, validation and error responses, fixture integrity, timing fields, and the frontend/API contract.

Run the complete real-OCR fixture evaluation:

```powershell
python tests\run_ocr_extraction.py
```

Expected result: all eight fixtures pass their intended outcome. The set includes a clean match, brand-case variation, incorrect warning case, ABV mismatch, class/type typo requiring review, net-contents mismatch, skew, and glare.

## API

- `GET /health` — process liveness.
- `GET /ready` — local OCR, optional provider, and Tesseract readiness.
- `POST /verify` — one image in `file` plus JSON string in `application_data`.
- `POST /verify/batch` — images in `files` plus either one shared application object or one object per image.

Successful single responses include `filename`, `extracted`, `comparison`, and `processing_time_ms`. Batch responses include ordered per-file results and a `progress` summary with total processing time.

## Render deployment

The included `render.yaml` installs the package and starts the single FastAPI service. RapidOCR model files ship with the installed package, so the deployed app works without `LLM_API_KEY` or a Tesseract system package.

1. Push the latest commit to the repository connected to Render.
2. Wait for the build to install `rapidocr` and `onnxruntime`.
3. Confirm `/ready` reports `"status":"ready"` and `"local_ocr_available":true`.
4. Open the home page, choose **Try a working sample**, then **Verify label**.
5. Confirm all five sample fields pass and processing time is displayed.

The first request after a free-tier cold start can include platform startup delay. The UI reports backend processing separately from total browser request time so that distinction is visible.

## Known prototype boundaries

- No COLA integration, identity system, persistent storage, or production retention policy.
- The five sample fields are not a complete beverage-specific legal review. Producer address, country-of-origin rules, font size, contrast, placement, and bold styling remain agent checks.
- Batch UI applies one application to all selected images; the API additionally supports one application object per file.
- Generated fixtures are deterministic regression inputs, not a substitute for evaluation on a representative set of real bottle photographs.
