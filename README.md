# Label Verify

Label Verify is a proof-of-concept web application that helps a reviewer read and compare alcohol beverage label information. It can:

- scan a label and transcribe five important fields;
- compare one or more labels with approved application values; and
- report each field as `pass`, `fail`, or `needs_review`.

Project documentation: [Approach, decisions, and assumptions](APPROACH_AND_DECISIONS.txt)

Live demo: <https://label-verify-p15v.onrender.com/>

> **Demo performance:** The public demo uses Render's free hosting tier, so the
> service may need extra time to start after it has been inactive. Accurate mode
> also uses Gemini through its free API tier, which can add provider latency or
> rate-limit delays. Once the Render service is awake.

> Label Verify is a review aid, not an automated compliance decision. A reviewer must confirm the image, warning styling, and requirements outside the prototype's five fields.

## Quick start

Requirements:

- Python 3.11 or newer
- Git

From the repository root, run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/>.

The frontend and API are served together, so there is no separate frontend installation. Fast mode uses the bundled local OCR models and does not require an API key.

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Try the application

1. Open the application and select **Try a working sample**.
2. Choose **Compare with application** or **Scan only**.
3. Select a reading mode:
   - **Fast** runs OCR locally on the application server.
   - **Accurate** uses a configured AI vision provider.
4. Select **Verify label** or **Scan label**.
5. Review the detected values and any fields marked for manual review.

You can also upload your own PNG, JPEG, WEBP, or GIF image up to 25 MB. Selecting multiple images starts a batch comparison.

## Optional: enable Accurate mode

Fast mode works immediately after installation. Accurate mode requires either Gemini or OpenAI credentials.

For Gemini, set:

```powershell
$env:GEMINI_API_KEY="your_key"
$env:GEMINI_MODEL="gemini-3.5-flash"  # optional; this is the default
```

Or use OpenAI:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:VISION_MODEL="gpt-5.4-mini"       # optional; this is the default
```

Restart the server after changing environment variables. See [`backend/.env.example`](backend/.env.example) for all available settings. When both provider keys are present, set `VISION_PROVIDER` to `gemini` or `openai` to choose one explicitly.

The default Gemini configuration was selected because it is available for this
free-tier demonstration. When running the application locally, you can try a
different Gemini model available to your API key by changing `GEMINI_MODEL`:

```powershell
$env:GEMINI_MODEL="another_supported_model"
python -m uvicorn app.main:app --reload
```

A faster or higher-capability provider model may improve response time or results
for difficult images, although availability, speed, quotas, and cost depend on
the provider. Gemini still runs through its API when the application is hosted
locally; only Fast mode performs OCR entirely on the application machine.

Accurate mode sends the uploaded image to the selected provider. Provider cost, retention, and data-use terms should be reviewed before using sensitive or production data.

## Run the tests

With the virtual environment active:

```powershell
cd backend
python -m pytest -q
```

To run the real local-OCR fixture evaluation:

```powershell
python tests\run_ocr_extraction.py
```

The test suite covers the API, comparison rules, provider integrations, local OCR, generated fixtures, error handling, and the frontend/backend contract.

## Project structure

```text
label-verify/
|-- backend/
|   |-- app/                 FastAPI routes, extraction, and comparison logic
|   |-- tests/               Automated tests and generated label fixtures
|   `-- pyproject.toml       Python package and dependencies
|-- frontend/                HTML, CSS, JavaScript, and sample label
|-- APPROACH_AND_DECISIONS.txt
|-- render.yaml              Render deployment configuration
`-- README.md
```

## API summary

| Method | Endpoint        | Purpose                                  |
| ------ | --------------- | ---------------------------------------- |
| `GET`  | `/health`       | Basic process health check               |
| `GET`  | `/ready`        | Reports available OCR and vision modes   |
| `POST` | `/scan`         | Extracts fields from one image           |
| `POST` | `/verify`       | Compares one image with application data |
| `POST` | `/verify/batch` | Compares multiple images                 |

Interactive API documentation is available at <http://127.0.0.1:8000/docs> while the server is running.

## Deploy with Render

The repository includes a Render Blueprint in `render.yaml`.

1. Push the repository to GitHub or another supported Git provider.
2. In Render, create a new Blueprint and select the repository.
3. Add `GEMINI_API_KEY` if Accurate mode should be enabled.
4. Deploy and confirm that `/ready` returns `"status": "ready"`.

Fast mode is self-contained because the RapidOCR ONNX model files are included in the repository.

The free Render service can have a cold-start delay after inactivity. This hosting
startup time is separate from the processing time reported by the backend and
from any additional Gemini free-tier delay in Accurate mode.

## Prototype limits

- The application has no login, database, COLA integration, or long-term file storage.
- It checks five fields only: brand name, class/type, alcohol content, net contents, and government warning.
- OCR cannot reliably confirm visual rules such as font size, bold text, contrast, or placement.
- The included fixtures are repeatable test inputs, not a substitute for testing representative real-world bottle photographs.
- A human reviewer remains responsible for the final compliance decision.
