document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();

  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  const errorMessage = document.getElementById('error-message');
  const errorText = document.querySelector('.error-text');
  const selectedFiles = document.getElementById('selected-files');
  const filesList = document.getElementById('files-list');
  const analyzeButtonContainer = document.getElementById('analyze-button-container');
  const analyzeButton = document.getElementById('analyze-button');
  const loadingSection = document.getElementById('loading-section');
  const errorSection = document.getElementById('error-section');
  const errorDetails = document.getElementById('error-details');
  const resultsSection = document.getElementById('results-section');
  const resultsList = document.getElementById('results-list');
  const processedCount = document.getElementById('processed-count');

  uploadArea.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files);
    if (files.length === 0) return;

    errorMessage.classList.add('hidden');
    selectedFiles.classList.remove('hidden');
    analyzeButtonContainer.classList.remove('hidden');
    filesList.innerHTML = "";

    files.forEach(file => {
      const item = document.createElement('div');
      item.textContent = `📄 ${file.name}`;
      filesList.appendChild(item);
    });
  });

  analyzeButton.addEventListener('click', async () => {
    const files = Array.from(fileInput.files);
    if (files.length === 0) return;

    // UI toggle
    loadingSection.classList.remove('hidden');
    document.getElementById('upload-section').classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    const formData = new FormData();
    files.forEach(file => {
      if (file.name.endsWith('.docx')) {
        formData.append('documents', file);
      }
    });

    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();

      loadingSection.classList.add('hidden');

      if (result.success) {
        resultsSection.classList.remove('hidden');
        resultsList.innerHTML = "";
        processedCount.textContent = `Processed ${result.processed_count} file(s).`;

        Object.entries(result.documents).forEach(([topic, summary]) => {
          const div = document.createElement('div');
          div.className = 'result-item';

          const h4 = document.createElement('h4');
          h4.textContent = topic;

          const p = document.createElement('p');
          p.textContent = summary;

          div.appendChild(h4);
          div.appendChild(p);
          resultsList.appendChild(div);
        });
      } else {
        throw new Error(result.error || "Unknown error");
      }
    } catch (err) {
      loadingSection.classList.add('hidden');
      errorSection.classList.remove('hidden');
      errorDetails.textContent = err.message;
    }
  });

  document.getElementById('try-again-button').addEventListener('click', () => {
    window.location.reload();
  });

  document.getElementById('new-analysis-button').addEventListener('click', () => {
    window.location.reload();
  });
});
