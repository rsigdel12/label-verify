# Label Verify Prototype

This FastAPI prototype accepts an alcohol-label image plus submitted application data, extracts five key fields with a vision-capable model, compares printed values against the application, and returns per-field `pass`, `fail`, or `needs_review` results.

Images are processed in memory and are not retained. A local Tesseract fallback is available when its system binary is installed, but a vision provider is the recommended extraction path for deployment and difficult images.

## Structure

- `backend/` - FastAPI application, comparison logic, and tests
- `frontend/` - accessible HTML/CSS/JS application form, uploader, and results view

## Local development

1. Create and activate a virtual environment.
2. Install dependencies:

   `cd backend && python -m pip install -e .`

3. Copy the environment template (PowerShell shown):

   `Copy-Item .env.example .env`

4. Set your provider variables in the process environment:
   - `LLM_API_KEY`
   - `LLM_BASE_URL` (default OpenAI-compatible endpoint)
   - `LLM_MODEL`

5. Start the API:

   `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

6. Serve the frontend locally:

   `cd ../frontend && python -m http.server 5500`

7. Open the frontend at `http://127.0.0.1:5500`.

## API summary

- `GET /health` - liveness check
- `GET /ready` - reports whether a provider or Tesseract is configured
- `POST /verify` - single upload and comparison
- `POST /verify/batch` - multiple uploads with per-file failures and capped concurrency

`/verify` expects multipart form data with an image in `file` and a JSON object string in `application_data`. Application values are `expected`; values extracted from the image are `actual`.

Successful responses also include `processing_time_ms`, measured inside the API. The frontend displays this separately from total browser request time so provider work is not confused with network latency.

The vision-provider request has a five-second timeout aligned with the stakeholder target. The frontend flags processing results that exceed five seconds; Render cold starts and total network time are reported separately.

## Validation and fixtures

Run the backend suite with `cd backend && python -m pytest -q`.

The suite covers matcher behavior, semantic ABV and metric-volume equivalence, government-warning case and wording, real multipart API requests, corrupt or unsupported images, unavailable extraction services, batch partial failures, fixture consistency, and the frontend/API contract.

The eight generated fixtures in `backend/tests/fixtures/` contain realistic distilled-spirits fields and the full federal government health warning. Each JSON sidecar separates `label_data`, `application_data`, and `expected_statuses`. Skew and glare are applied after rendering the text, so they exercise the content under test.

## Environment

Copy `backend/.env.example` to `backend/.env` and set your provider configuration before enabling live extraction. The app does not load `.env` files by itself unless the launcher does so.

## Render deployment

The repository includes a Render Blueprint. `LLM_API_KEY` is declared with `sync: false`, so it must be entered as a secret in the Render dashboard and must never be committed.

1. Set `LLM_API_KEY` in the Render service's Environment page.
2. Deploy the latest commit and confirm `/health` returns `{"status":"ok"}`.
3. Check `/ready`; `vision_provider_configured` should be `true`.
4. Submit `fixture_01_clean_match.png` with `application_data` from its JSON sidecar.
5. Inspect Render logs if the provider rejects the credentials, model, or outbound request.

## Trade-offs and constraints

- There is no persistent storage, COLA integration, or enterprise identity integration.
- The provider call uses an OpenAI-compatible Chat Completions request. Other provider schemas require an adapter.
- The five-field response verifies warning wording and capitalization, but it cannot prove that `GOVERNMENT WARNING:` is bold. Visual-format classification is a next step.
- Mock latency is not a production benchmark. Measure representative requests on Render before claiming the five-second target.
