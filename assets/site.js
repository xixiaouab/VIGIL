const copyButton = document.querySelector("[data-copy-bib]");
const bibtex = document.querySelector("#bibtex");

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch {
      copyButton.textContent = "Select text";
    }
  });
}
