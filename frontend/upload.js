const STANDARD_WARNING =
  "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not " +
  "drink alcoholic beverages during pregnancy because of the risk of birth defects. " +
  "(2) Consumption of alcoholic beverages impairs your ability to drive a car or " +
  "operate machinery, and may cause health problems.";

const SAMPLE_APPLICATION = {
  brand_name: "Acme Spirits",
  class_type: "Kentucky Straight Bourbon Whiskey",
  alcohol_content: "45% Alc. by Vol.",
  net_contents: "750 mL",
  warning_statement: STANDARD_WARNING,
};

const MAX_FILE_SIZE = 25 * 1024 * 1024;
const ALLOWED_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);

const FIELD_LABELS = {
  brand_name: "Brand name",
  class_type: "Class or type",
  alcohol_content: "Alcohol content",
  net_contents: "Net contents",
  warning_statement: "Government warning",
};

const STATUS_LABELS = {
  pass: "Pass",
  fail: "Fail",
  needs_review: "Needs review",
};

const form = document.getElementById("verifyForm");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const filePreview = document.getElementById("filePreview");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFileButton = document.getElementById("removeFileButton");
const submitButton = document.getElementById("submitButton");
const buttonLabel = submitButton.querySelector(".button-label");
const sampleButton = document.getElementById("sampleButton");
const loading = document.getElementById("loading");
const loadingElapsed = document.getElementById("loadingElapsed");
const errorBanner = document.getElementById("errorBanner");
const errorMessage = document.getElementById("errorMessage");
const resultsSection = document.getElementById("resultsSection");
const resultSummary = document.getElementById("resultSummary");
const resultsHeading = document.getElementById("resultsHeading");
const resultCounts = document.getElementById("resultCounts");
const summaryIcon = document.getElementById("summaryIcon");
const serverTime = document.getElementById("serverTime");
const roundTripTime = document.getElementById("roundTripTime");
const resultFields = document.getElementById("resultFields");
const verifyAnotherButton = document.getElementById("verifyAnotherButton");

let selectedFile = null;
let previewUrl = null;
let timerId = null;

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "Not available";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(2)} seconds`;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function hideError() {
  errorBanner.classList.add("hidden");
  errorMessage.textContent = "";
}

function showError(message, requestTimeMs = null) {
  const timing = Number.isFinite(requestTimeMs)
    ? ` The request ended after ${formatDuration(requestTimeMs)}.`
    : "";
  errorMessage.textContent = `${message}${timing}`;
  errorBanner.classList.remove("hidden");
  errorBanner.focus({ preventScroll: true });
  errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearPreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function clearSelectedFile() {
  selectedFile = null;
  fileInput.value = "";
  clearPreviewUrl();
  previewImage.removeAttribute("src");
  filePreview.classList.add("hidden");
  uploadArea.classList.remove("hidden", "is-dragging");
}

function selectFile(file) {
  hideError();
  if (!file) return;

  if (!ALLOWED_TYPES.has(file.type)) {
    clearSelectedFile();
    showError("Choose a PNG, JPEG, WEBP, or GIF image.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    clearSelectedFile();
    showError("The selected image exceeds the 25 MB upload limit.");
    return;
  }

  selectedFile = file;
  clearPreviewUrl();
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  fileName.textContent = file.name;
  fileSize.textContent = `${formatFileSize(file.size)} · ${file.type}`;
  uploadArea.classList.add("hidden");
  filePreview.classList.remove("hidden");
}

function applicationPayload() {
  return {
    brand_name: document.getElementById("brandName").value.trim(),
    class_type: document.getElementById("classType").value.trim(),
    alcohol_content: document.getElementById("alcoholContent").value.trim(),
    net_contents: document.getElementById("netContents").value.trim(),
    warning_statement: document.getElementById("warningStatement").value.trim(),
  };
}

function loadSampleApplication() {
  for (const [field, value] of Object.entries(SAMPLE_APPLICATION)) {
    const input = form.elements.namedItem(field);
    if (input) input.value = value;
  }
  document.getElementById("brandName").focus();
}

function startLoadingTimer(startedAt) {
  loadingElapsed.textContent = "Elapsed: 0.0 seconds";
  timerId = window.setInterval(() => {
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    loadingElapsed.textContent = `Elapsed: ${elapsedSeconds.toFixed(1)} seconds`;
  }, 100);
}

function stopLoadingTimer() {
  if (timerId !== null) {
    window.clearInterval(timerId);
    timerId = null;
  }
}

function setLoading(isLoading, startedAt = null) {
  submitButton.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Verifying label..." : "Verify label";
  loading.classList.toggle("hidden", !isLoading);
  if (isLoading) startLoadingTimer(startedAt);
  else stopLoadingTimer();
}

function createValueBlock(label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value ?? "Not detected";
  wrapper.append(term, description);
  return wrapper;
}

function renderComparison(comparison, processingTimeMs, totalRequestMs) {
  const counts = { pass: 0, fail: 0, needs_review: 0 };
  resultFields.replaceChildren();

  for (const [field, entry] of Object.entries(comparison)) {
    const status = STATUS_LABELS[entry?.status] ? entry.status : "needs_review";
    counts[status] += 1;

    const row = document.createElement("article");
    row.className = "field-result";

    const name = document.createElement("div");
    name.className = "field-name";
    name.textContent = FIELD_LABELS[field] || field.replaceAll("_", " ");

    const values = document.createElement("dl");
    values.className = "field-values";
    values.append(
      createValueBlock("Expected", entry?.expected),
      createValueBlock("Detected", entry?.actual),
    );

    const badge = document.createElement("span");
    badge.className = `badge ${status}`;
    badge.textContent = STATUS_LABELS[status];

    row.append(name, values, badge);
    resultFields.append(row);
  }

  let overallStatus = "pass";
  let heading = "All fields verified";
  let icon = "OK";
  if (counts.fail > 0) {
    overallStatus = "fail";
    heading = "Differences found";
    icon = "X";
  } else if (counts.needs_review > 0) {
    overallStatus = "needs_review";
    heading = "Manual review needed";
    icon = "!";
  }

  resultSummary.className = `result-summary ${overallStatus}`;
  resultsHeading.textContent = heading;
  summaryIcon.textContent = icon;
  resultCounts.textContent =
    `${counts.pass} passed, ${counts.fail} failed, ` +
    `${counts.needs_review} need review.`;
  serverTime.textContent = formatDuration(processingTimeMs);
  serverTime.classList.toggle("over-target", processingTimeMs > 5000);
  serverTime.title = processingTimeMs > 5000 ? "Exceeded the five-second target" : "";
  roundTripTime.textContent = formatDuration(totalRequestMs);
  resultsSection.classList.remove("hidden");
  resultsSection.focus({ preventScroll: true });
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function responseError(response) {
  let message = `The server returned HTTP ${response.status}.`;
  if (response.headers.get("content-type")?.includes("application/json")) {
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // Retain the status-based fallback when the error body is malformed.
    }
  }
  return message;
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
removeFileButton.addEventListener("click", () => {
  clearSelectedFile();
  fileInput.focus();
});
sampleButton.addEventListener("click", loadSampleApplication);

for (const eventName of ["dragenter", "dragover"]) {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.remove("is-dragging");
  });
}

uploadArea.addEventListener("drop", (event) => {
  selectFile(event.dataTransfer.files[0]);
});

verifyAnotherButton.addEventListener("click", () => {
  clearSelectedFile();
  resultsSection.classList.add("hidden");
  hideError();
  fileInput.focus();
  uploadArea.scrollIntoView({ behavior: "smooth", block: "center" });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();

  if (!form.reportValidity()) return;
  if (!selectedFile) {
    showError("Choose a label image before starting verification.");
    fileInput.focus();
    return;
  }

  const requestBody = new FormData();
  requestBody.append("file", selectedFile);
  requestBody.append("application_data", JSON.stringify(applicationPayload()));

  resultsSection.classList.add("hidden");
  const startedAt = performance.now();
  setLoading(true, startedAt);

  try {
    const response = await fetch("/verify", { method: "POST", body: requestBody });
    const totalRequestMs = performance.now() - startedAt;

    if (!response.ok) {
      showError(await responseError(response), totalRequestMs);
      return;
    }

    const data = await response.json();
    if (!data.comparison) {
      showError("The server response did not include comparison results.", totalRequestMs);
      return;
    }
    renderComparison(data.comparison, data.processing_time_ms, totalRequestMs);
  } catch (error) {
    showError(
      `The request could not reach the verification service: ${error.message}`,
      performance.now() - startedAt,
    );
  } finally {
    setLoading(false);
  }
});

window.addEventListener("beforeunload", clearPreviewUrl);
