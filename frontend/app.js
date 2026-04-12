const apiBaseEl = document.getElementById("apiBase");
const thresholdEl = document.getElementById("threshold");
const rowsJsonEl = document.getElementById("rowsJson");
const statusEl = document.getElementById("status");
const classLabelEl = document.getElementById("classLabel");
const classProbEl = document.getElementById("classProb");
const regValueEl = document.getElementById("regValue");

function setStatus(message) {
  statusEl.textContent = message;
}

function makeTemplateRows(columns, lookback) {
  const rows = [];
  for (let i = 0; i < lookback; i += 1) {
    const row = {};
    columns.forEach((col) => {
      row[col] = 0;
    });
    rows.push(row);
  }
  return rows;
}

async function loadMetadata() {
  const base = apiBaseEl.value.trim();
  setStatus("Loading metadata...");

  try {
    const response = await fetch(`${base}/metadata`);
    if (!response.ok) {
      throw new Error(`Metadata request failed: ${response.status}`);
    }

    const data = await response.json();
    const cols = data.classifier.feature_columns;
    const lookback = data.classifier.lookback;

    rowsJsonEl.value = JSON.stringify(makeTemplateRows(cols, lookback), null, 2);
    setStatus(`Loaded metadata. lookback=${lookback}, columns=${cols.length}`);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
}

async function runForecast() {
  const base = apiBaseEl.value.trim();
  setStatus("Running forecast...");

  let rows;
  try {
    rows = JSON.parse(rowsJsonEl.value);
    if (!Array.isArray(rows)) {
      throw new Error("Rows JSON must be an array");
    }
  } catch (error) {
    setStatus(`JSON parse error: ${error.message}`);
    return;
  }

  const payload = {
    rows,
    classifier_threshold: Number(thresholdEl.value),
  };

  try {
    const response = await fetch(`${base}/predict/all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Predict request failed: ${response.status}`);
    }

    classLabelEl.textContent = data.classifier.label;
    classProbEl.textContent = data.classifier.probability_of_rain.toFixed(4);
    regValueEl.textContent = data.regressor.predicted_precipitation_mm.toFixed(4);
    setStatus("Forecast complete.");
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
}

document.getElementById("loadMeta").addEventListener("click", loadMetadata);
document.getElementById("runPredict").addEventListener("click", runForecast);

loadMetadata();
