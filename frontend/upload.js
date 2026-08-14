const fileInput = document.getElementById("fileInput");
const submitButton = document.getElementById("submitButton");
const resultEl = document.getElementById("result");
const loadingEl = document.getElementById("loading");
const uploadArea = document.getElementById("uploadArea");

// Drag & drop support
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = "#764ba2";
  uploadArea.style.background = "#f0f2ff";
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.style.borderColor = "#667eea";
  uploadArea.style.background = "#f8f9ff";
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = "#667eea";
  uploadArea.style.background = "#f8f9ff";
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
  }
});

function renderComparison(comparison) {
  let passCount = 0;
  let failCount = 0;
  let reviewCount = 0;

  const rows = Object.entries(comparison)
    .map(([field, entry]) => {
      const status = entry?.status || "needs_review";
      const badgeClass = status.toLowerCase();
      
      if (status === "pass") passCount++;
      else if (status === "fail") failCount++;
      else reviewCount++;

      return `
        <div class="field-result">
          <div class="field-header">
            <strong>${field.replace(/_/g, " ")}</strong>
            <span class="badge ${badgeClass}">${status}</span>
          </div>
          <div class="field-values">
            <div><span>Expected:</span> ${entry?.expected ?? "—"}</div>
            <div><span>Actual:</span> ${entry?.actual ?? "—"}</div>
          </div>
        </div>
      `;
    })
    .join("");

  const summaryStatus = failCount > 0 ? "❌ Failed" : (reviewCount > 0 ? "⚠️ Review Needed" : "✅ All Verified");
  const summary = `
    <div class="${failCount > 0 ? "error-message" : "success-summary"}">
      <strong>${summaryStatus}</strong> — 
      ${passCount} verified, ${failCount} failed, ${reviewCount} needs review
    </div>
  `;

  resultEl.innerHTML = summary + rows;
}

function showError(message) {
  resultEl.innerHTML = `<div class="error-message">
    <strong>⚠️ Error:</strong> ${escapeHtml(message)}
  </div>`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

submitButton.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    showError("Please choose an image file first.");
    return;
  }

  // Show loading state
  submitButton.disabled = true;
  submitButton.textContent = "Verifying...";
  loadingEl.classList.remove("hidden");
  resultEl.innerHTML = "";

  const payload = {
    brand_name: "Acme Spirits",
    class_type: "Vodka",
    alcohol_content: "40% vol",
    net_contents: "750 ml",
    warning_statement: "Contains sulfites.",
  };

  const formData = new FormData();
  formData.append("file", file);
  formData.append("application_data", JSON.stringify(payload));

  try {
    const response = await fetch("/verify", {
      method: "POST",
      body: formData,
    });

    // Check if response is OK (200-299 status)
    if (!response.ok) {
      const contentType = response.headers.get("content-type");
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      
      // Try to parse error details from response
      if (contentType && contentType.includes("application/json")) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          // If JSON parsing fails, use the status
        }
      }
      
      showError(errorMessage);
      return;
    }

    // Parse response as JSON
    const data = await response.json();

    if (!data.comparison) {
      showError("Unexpected response format: missing comparison data");
      console.error("Full response:", data);
      return;
    }

    renderComparison(data.comparison);
  } catch (error) {
    showError(`Request failed: ${error.message}`);
    console.error("Full error:", error);
  } finally {
    // Hide loading state
    submitButton.disabled = false;
    submitButton.textContent = "Verify Label";
    loadingEl.classList.add("hidden");
  }
});
