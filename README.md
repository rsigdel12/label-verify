# Label Verify

A standalone proof of concept for comparing alcohol label artwork with an approved application. Reviewers upload one or more label images, enter the application values, and receive field-level `pass`, `fail`, or `needs_review` results with measured processing time.

Deployed application: <https://label-verify-p15v.onrender.com/>

## What the prototype covers

| Stakeholder need | Implementation |
| --- | --- |
| Obvious workflow for mixed technical comfort levels | One two-step screen, drag-and-drop upload, plain-language statuses, keyboard labels, responsive layout, and a one-click working sample |
| Explicit speed/accuracy choice | **Fast** runs one bounded local OCR pass and targets under five seconds; **Accurate** uses contextual AI vision and may take 5–15 seconds |
| Compare label artwork with an application | The application form is the expected record; OCR text from the image is the actual value |
| Required sample fields | Brand name, class/type, alcohol content, net contents, and complete government warning |
| Faster application entry | The standard TTB government warning is prefilled and remains editable when an approved record differs |
| Consistent class/type entry | A common TTB-informed alcohol-type enum supplies form suggestions and backend category matching while still allowing specific designations |
| Human judgment for near matches | Brand casing/spacing is tolerant; a close but nonidentical regulated class/type is `needs_review` rather than silently approved |
| Exact warning language and capitalization | Warning comparison preserves case, punctuation, spelling, numbering, and order; visual/OCR whitespace is ignored |
| Imperfect photos | Test coverage includes 7-degree skew and simulated glare |
| Peak-season batch work | Multi-select in the same uploader calls the batch API and shows results per file |
| Restricted outbound network | Local RapidOCR + ONNX Runtime remains available when a vision API is not configured or cannot be reached |
| Prototype privacy | Uploads are validated and not retained by the app; vision-mode images are sent to the configured API provider for processing |
| Useful error handling | Unsupported, empty, oversized, corrupt, unavailable, timeout, and per-file batch failures return actionable messages |

The app intentionally assists rather than replaces the compliance agent. The warning's wording and capitalization are checked, but OCR text alone cannot prove that the `GOVERNMENT WARNING:` heading is visually bold. The reviewer must confirm styling and any requirements outside the five prototype fields.

### Fixture validity

The generated distilled-spirits fixtures follow the current TTB examples for the fields in scope: brand, class/type, and alcohol content appear together; net contents uses the accepted `750 mL` form; and the standard warning heading is uppercase and bold while the warning body is regular weight. The domestic label also includes `DISTILLED AND BOTTLED BY ACME SPIRITS — LOUISVILLE, KENTUCKY`; country of origin is not applicable to this domestic example. See TTB's [mandatory distilled-spirits information](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-brand-label), [health warning](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-health-warning), and [net contents](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-net-contents) guidance.

## Architecture and decisions

- `backend/app/` — FastAPI routes, validation, local/provider extraction, and comparison rules.
- `frontend/` — dependency-free HTML, CSS, and JavaScript served by FastAPI.
- `backend/tests/fixtures/` — eight generated label images with JSON application data and intended results.
- `frontend/assets/sample-label.png` — self-contained sample used by the live demonstration.

Each request selects its extractor explicitly. **Fast** is the default and keeps the image on the application server for one local OCR pass. **Accurate** sends a 1600-pixel JPEG to `gpt-5.4-mini` through the Responses API with original-detail vision and a strict five-field JSON schema. The prompt requires visible transcription rather than filling likely warning text from memory. The request is not stored by this application and sets `store: false`; the provider attempt has a 15-second total deadline. The two paths are not run sequentially, so a failed Accurate request does not add local OCR time afterward.

Fast mode uses RapidOCR locally. Images are normalized with Pillow, limited to 20 megapixels, corrected using phone-camera EXIF orientation, and resized to a 720-pixel bound. The PP-OCRv6 tiny detector is capped at 384 pixels to fit the Render instance. Its shared ONNX sessions are preloaded at application startup so model initialization is not charged to the first Fast request. The internal `auto` compatibility mode can add a targeted contrast-enhanced retry, but it is intentionally not shown in the UI because its latency is less predictable. A system Tesseract installation remains a final development fallback.

RapidOCR and the bundled PP-OCRv6 model assets are used under the Apache-2.0 license.

### Comparison rules

- Brand name: case-insensitive fuzzy comparison with a 90% threshold.
- Class/type: common spirits, wine, malt-beverage, cider, and sake types are enum-classified. A selected broad type can match its detected subtype; conflicting subtypes fail; uncertain close text requires review.
- Alcohol content: percent, `ABV`, `Alcohol by Volume`, decimal-comma, and proof formats are normalized, so `40%` and `80 Proof` are equivalent.
- Net contents: mL, cL, liters, and fluid ounces are normalized; common OCR unit confusions such as `m1` are corrected.
- Government warning: exact wording passes; whitespace caused by OCR segmentation is ignored; a high-similarity transcription with a verified uppercase heading requires review instead of producing an unreliable approval or false failure.
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

No environment variable is required for Fast local OCR. To enable Accurate AI vision in the selector, set:

- `OPENAI_API_KEY`
- `VISION_BASE_URL` (defaults to `https://api.openai.com/v1`)
- `VISION_MODEL` (defaults to `gpt-5.4-mini`)
- `VISION_REASONING_EFFORT` (defaults to `none` for latency)
- `EXTRACTION_MODE` (defaults to `local` for older API clients that omit the per-request mode)

The legacy `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` names remain accepted for compatibility.

## Test and evaluate

Run the automated suite:

```powershell
cd label-verify\backend
python -m pytest -q
```

The suite contains 52 tests, including per-request mode selection, vision request schema/failure behavior, and real local OCR through both single and batch multipart endpoints. It also covers startup loading, enum classification, alternate ABV and net-content formats, warning uncertainty, validation and error responses, fixture integrity, timing fields, and the frontend/API contract.

Run the complete real-OCR fixture evaluation:

```powershell
python tests\run_ocr_extraction.py
```

Fast mode completes all eight fixtures in under one second each on the reference development machine. Seven match their intended outcome; the 7-degree skew fixture conservatively returns `needs_review` for one warning word instead of approving an imperfect transcription. Accurate mode is intended for that kind of difficult image. The set also includes a clean match, brand-case variation, incorrect warning case, ABV mismatch, class/type typo requiring review, net-contents mismatch, and glare.

## API

- `GET /health` — process liveness.
- `GET /ready` — local OCR, optional provider, and Tesseract readiness.
- `POST /verify` — one image in `file`, JSON string in `application_data`, and optional `extraction_mode` (`local` or `vision`).
- `POST /verify/batch` — images in `files`, either one shared application object or one object per image, and optional shared `extraction_mode`.

Successful single responses include `filename`, `extracted`, `comparison`, and `processing_time_ms`. Batch responses include ordered per-file results and a `progress` summary with total processing time.

## Render deployment

The included `render.yaml` installs the package and starts the single FastAPI service. RapidOCR model files ship with the installed package, so the deployed app works without `LLM_API_KEY` or a Tesseract system package.

1. Push the latest commit to the repository connected to Render.
2. Add `OPENAI_API_KEY` as a secret environment variable to enable Accurate mode. `VISION_MODEL=gpt-5.4-mini` is optional because it is the default.
3. Wait for the build to install `rapidocr` and `onnxruntime`.
4. Confirm `/ready` reports `"status":"ready"`, `"local_ocr_available":true`, and `"vision_provider_configured":true` when the key is present.
5. Open the home page, choose **Try a working sample**, select a reading mode, then **Verify label**.
6. Confirm all five sample fields pass and processing time is displayed.

The first request after a free-tier cold start can include platform startup delay. The UI reports backend processing separately from total browser request time so that distinction is visible.

## Known prototype boundaries

- No COLA integration, identity system, persistent storage, or production retention policy.
- The five sample fields are not a complete beverage-specific legal review. Producer address, country-of-origin rules, font size, contrast, placement, and bold styling remain agent checks.
- Batch UI applies one application to all selected images; the API additionally supports one application object per file.
- Generated fixtures are deterministic regression inputs, not a substitute for evaluation on a representative set of real bottle photographs.
