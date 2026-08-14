const fileInput = document.getElementById("fileInput");
const submitButton = document.getElementById("submitButton");
const resultEl = document.getElementById("result");

submitButton.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    resultEl.textContent = "Please choose a file first.";
    return;
  }
  function renderComparison(comparison) {
    const rows = Object.entries(comparison)
      .map(([field, entry]) => {
        const status = entry?.status || "needs_review";
        const badgeClass = status.toLowerCase();
        return `
          <div class="field-result">
            <div class="field-header">
              <strong>${field.replace("_", " ")}</strong>
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

    resultEl.innerHTML = rows || "<p>No comparison data returned.</p>";
  }

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

    const data = await response.json();
    if (!data.comparison) {
      resultEl.textContent = JSON.stringify(data, null, 2);
      return;
    }

    renderComparison(data.comparison);
  } catch (error) {
    resultEl.textContent = `Request failed: ${error.message}`;
  }
});
