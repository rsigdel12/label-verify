# Label Verify Prototype

This project contains a minimal FastAPI prototype for AI-powered label verification. The app accepts an image plus submitted label data, extracts the key fields using a vision-capable LLM, compares them against the application values, and returns per-field pass/fail/needs-review status.

Note: this environment does not include cloud deployment credentials, so the app was validated locally against a mock OpenAI-compatible provider to verify the full request path. The deployment steps below are included for a Render/Vercel/Railway account as a ready-to-run checklist.

## Structure

- `backend/` - FastAPI application and tests
- `frontend/` - lightweight HTML/JS uploader for local testing

## Local development

1. Create and activate a virtual environment.
2. Install dependencies:

   `cd backend && python -m pip install -e .`

3. Copy the environment template:

   `cp .env.example .env`

4. Set your provider variables in `backend/.env`:
   - `LLM_API_KEY`
   - `LLM_BASE_URL` (default OpenAI-compatible endpoint)
   - `LLM_MODEL`

5. Start the API:

   `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

6. Serve the frontend locally:

   `cd ../frontend && python -m http.server 5500`

7. Open the frontend at `http://127.0.0.1:5500` and submit a label image.

## API summary

- `GET /health` — liveness check
- `POST /verify` — single upload and comparison
- `POST /verify/batch` — multiple uploads with concurrency cap (15)

## Validation notes

The prototype has been validated locally with:

- matcher unit tests (`4 passed`)
- health check on `http://127.0.0.1:8000/health`
- single-file `/verify` request returning pass/fail/needs-review data
- batch `/verify/batch` request with multiple uploads and aggregated progress
- frontend served at `http://127.0.0.1:5500` and CORS validated for the upload endpoint

The end-to-end request completed in roughly `0.42` seconds in the local mock-provider path, which remains comfortably under the intended 5-second budget.

## Environment

Copy `backend/.env.example` to `backend/.env` and set your provider configuration before enabling live extraction.

## Deployment checklist

This project is ready to deploy to a platform such as Render, Railway, or a simple Vercel static frontend plus a separate backend service.

Recommended deployment pattern:

1. Deploy the FastAPI backend as a web service with environment variables for `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
2. Deploy the frontend as static files and point the fetch URL to the deployed backend service.
3. Confirm the deployed `/health` route is reachable from a fresh browser/session.
4. Test `/verify` and `/verify/batch` against a live provider or mock endpoint.

## Trade-offs and constraints

- There is no persistent storage in this version; uploaded files are processed in memory only.
- There is no COLA or enterprise identity integration in the prototype.
- Extraction is coupled to a single LLM provider configuration; swapping providers requires changing the request format in `app/extraction/vision_client.py`.
