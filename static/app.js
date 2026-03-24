const previewForm = document.getElementById("previewForm");
const batchForm = document.getElementById("batchForm");
const previewStatus = document.getElementById("previewStatus");
const batchStatus = document.getElementById("batchStatus");
const previewImage = document.getElementById("previewImage");

previewForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  previewStatus.textContent = "Generating preview... this may take a little while.";
  previewImage.style.display = "none";

  const formData = new FormData(previewForm);
  const response = await fetch("/api/preview", { method: "POST", body: formData });

  if (!response.ok) {
    const err = await response.json();
    previewStatus.textContent = err.error || "Preview failed.";
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  previewImage.src = url;
  previewImage.style.display = "block";
  previewStatus.textContent = "Preview generated.";
});

batchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  batchStatus.textContent = "Processing batch export...";

  const formData = new FormData(batchForm);
  const response = await fetch("/api/batch", { method: "POST", body: formData });
  const payload = await response.json();

  if (!response.ok) {
    batchStatus.textContent = payload.error || "Batch export failed.";
    return;
  }

  const files = payload.exported.length
    ? payload.exported.map((f) => `• ${f}`).join("\n")
    : "No valid files were exported.";

  batchStatus.textContent = `Batch complete (${payload.count} images).\n${files}`;
});
