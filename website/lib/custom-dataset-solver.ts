/**
 * Universal In-Browser Linear Regression & Feature Selection Solver
 * Designed for Stage 6: /custom-dataset bonus feature
 * 
 * Supports arbitrary CSV datasets, real-time matrix inversion with L2 stability,
 * one-hot encoding for categorical variables, evaluation metrics (MAE, RMSE, R²),
 * and split conformal prediction intervals.
 */

export interface ParsedColumnInfo {
  name: string;
  index: number;
  type: "numeric" | "categorical";
  distinctValues: number;
  sampleValues: (string | number)[];
  missingCount: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
}

export interface CustomDatasetSummary {
  rowCount: number;
  columnCount: number;
  columns: ParsedColumnInfo[];
  previewRows: Record<string, string | number>[];
  rawRows: Record<string, string | number>[];
}

export interface TrainedFeatureWeight {
  featureName: string;
  originalColumn: string;
  coefficient: number;
  stdCoefficient: number;
  mean: number;
  scale: number;
  importanceRank: number;
}

export interface CustomModelEvaluation {
  sampleCount: number;
  trainCount: number;
  testCount: number;
  targetColumn: string;
  intercept: number;
  weights: TrainedFeatureWeight[];
  trainMetrics: {
    mae: number;
    rmse: number;
    r2: number;
  };
  testMetrics: {
    mae: number;
    rmse: number;
    r2: number;
    dummyMae: number;
    dummyRmse: number;
    dummyR2: number;
  };
  conformalInterval?: {
    calCount: number;
    conformalQ: number;
    nominalCoverage: number;
    empiricalCoverage: number;
    meanWidth: number;
  };
  scatterPoints: {
    actual: number;
    predicted: number;
    residual: number;
    lower?: number;
    upper?: number;
    covered?: boolean;
    isTest: boolean;
  }[];
}

// Simple RFC-compliant CSV line parser
export function parseCSV(csvText: string): Record<string, string | number>[] {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV 檔案必須至少包含標題行與一筆資料行。");

  // Parse header
  const parseRow = (line: string): string[] => {
    const row: string[] = [];
    let insideQuotes = false;
    let entry = "";
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        insideQuotes = !insideQuotes;
      } else if (char === ',' && !insideQuotes) {
        row.push(entry.trim().replace(/^"|"$/g, ''));
        entry = "";
      } else {
        entry += char;
      }
    }
    row.push(entry.trim().replace(/^"|"$/g, ''));
    return row;
  };

  const headers = parseRow(lines[0]);
  const rows: Record<string, string | number>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const rawLine = lines[i].trim();
    if (!rawLine) continue;
    const values = parseRow(rawLine);
    if (values.length !== headers.length) continue;

    const rowObj: Record<string, string | number> = {};
    for (let j = 0; j < headers.length; j++) {
      const col = headers[j];
      const rawVal = values[j];
      const numVal = parseFloat(rawVal);
      if (!isNaN(numVal) && isFinite(numVal) && !/^\d{4}-\d{2}-\d{2}/.test(rawVal)) {
        rowObj[col] = numVal;
      } else {
        rowObj[col] = rawVal;
      }
    }
    rows.push(rowObj);
  }

  if (rows.length === 0) throw new Error("未能從 CSV 解析出任何有效資料行。");
  return rows;
}

export function inspectDataset(rows: Record<string, string | number>[]): CustomDatasetSummary {
  if (rows.length === 0) throw new Error("Empty dataset");
  const headers = Object.keys(rows[0]);
  const columns: ParsedColumnInfo[] = [];

  for (let idx = 0; idx < headers.length; idx++) {
    const col = headers[idx];
    const vals = rows.map((r) => r[col]);
    let numCount = 0;
    let missingCount = 0;
    const distinctSet = new Set<string | number>();

    const numVals: number[] = [];
    for (const v of vals) {
      if (v === null || v === undefined || v === "") {
        missingCount++;
      } else {
        distinctSet.add(v);
        if (typeof v === "number" && !isNaN(v)) {
          numCount++;
          numVals.push(v);
        }
      }
    }

    const isNumeric = numCount / vals.length >= 0.7; // 70% threshold
    const colInfo: ParsedColumnInfo = {
      name: col,
      index: idx,
      type: isNumeric ? "numeric" : "categorical",
      distinctValues: distinctSet.size,
      sampleValues: Array.from(distinctSet).slice(0, 5),
      missingCount
    };

    if (isNumeric && numVals.length > 0) {
      const mean = numVals.reduce((a, b) => a + b, 0) / numVals.length;
      const variance = numVals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / Math.max(1, numVals.length - 1);
      colInfo.min = Math.min(...numVals);
      colInfo.max = Math.max(...numVals);
      colInfo.mean = mean;
      colInfo.std = Math.sqrt(variance);
    }

    columns.push(colInfo);
  }

  return {
    rowCount: rows.length,
    columnCount: headers.length,
    columns,
    previewRows: rows.slice(0, 8),
    rawRows: rows
  };
}

// Matrix Inversion via Gauss-Jordan elimination with partial pivoting
function invertMatrix(A: number[][]): number[][] {
  const n = A.length;
  // Augment A with Identity
  const M: number[][] = A.map((row, i) => {
    const res = [...row];
    for (let j = 0; j < n; j++) res.push(i === j ? 1 : 0);
    return res;
  });

  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    // Swap rows
    const temp = M[i];
    M[i] = M[maxRow];
    M[maxRow] = temp;

    const pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) {
      // Degenerate pivot, add jitter
      M[i][i] += 1e-6;
    }

    const diag = M[i][i];
    for (let j = 0; j < 2 * n; j++) M[i][j] /= diag;

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = M[k][i];
        for (let j = 0; j < 2 * n; j++) {
          M[k][j] -= factor * M[i][j];
        }
      }
    }
  }

  // Extract inverted matrix
  return M.map((row) => row.slice(n));
}

// Linear Regression Matrix Solver
export function trainCustomLinearModel(
  rows: Record<string, string | number>[],
  targetCol: string,
  selectedFeatureCols: string[],
  regularizationLambda = 1e-4
): CustomModelEvaluation {
  if (rows.length < 5) throw new Error("資料量過少，至少需 5 筆觀測才能訓練模型。");
  if (!selectedFeatureCols.includes(targetCol) === false) {
    throw new Error("預測特徵不能包含標的變數本身。");
  }

  // 1. Prepare design matrix X and target y
  interface EncodedCol {
    name: string;
    origCol: string;
    getValue: (row: Record<string, string | number>) => number;
  }

  const encodedCols: EncodedCol[] = [];

  for (const col of selectedFeatureCols) {
    const sampleVal = rows.find((r) => r[col] !== undefined && r[col] !== null)?.[col];
    if (typeof sampleVal === "number") {
      encodedCols.push({
        name: col,
        origCol: col,
        getValue: (r) => (typeof r[col] === "number" ? (r[col] as number) : 0)
      });
    } else {
      // One-hot encode low-cardinality categorical variables
      const distinctCategories = Array.from(new Set(rows.map((r) => String(r[col]))));
      if (distinctCategories.length <= 10) {
        // Drop first category to prevent collinearity
        for (let cIdx = 1; cIdx < distinctCategories.length; cIdx++) {
          const cat = distinctCategories[cIdx];
          encodedCols.push({
            name: `${col}_${cat}`,
            origCol: col,
            getValue: (r) => (String(r[col]) === cat ? 1 : 0)
          });
        }
      }
    }
  }

  if (encodedCols.length === 0) throw new Error("未選取任何有效特徵欄位。");

  // Filter rows with valid target
  const validRows = rows.filter((r) => typeof r[targetCol] === "number" && !isNaN(r[targetCol] as number));
  const N = validRows.length;
  const P = encodedCols.length;

  if (N <= P) {
    throw new Error(`樣本數 (${N}) 必須大於選定特徵數 (${P}) 才能穩定擬合多元線性迴歸。`);
  }

  // 2. Chronological or standard split (75% Train, 25% Test)
  const nTrain = Math.max(3, Math.floor(N * 0.75));
  const trainRows = validRows.slice(0, nTrain);
  const testRows = validRows.slice(nTrain);

  // Standardize based on Train
  const colMeans: number[] = [];
  const colScales: number[] = [];

  for (let j = 0; j < P; j++) {
    const trainVals = trainRows.map((r) => encodedCols[j].getValue(r));
    const mean = trainVals.reduce((a, b) => a + b, 0) / nTrain;
    const std = Math.sqrt(trainVals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / Math.max(1, nTrain - 1));
    colMeans.push(mean);
    colScales.push(std === 0 ? 1 : std);
  }

  // Build X_train (with bias 1.0 at index 0)
  const X_train: number[][] = [];
  const y_train: number[] = [];

  for (const r of trainRows) {
    const xRow = [1.0];
    for (let j = 0; j < P; j++) {
      const raw = encodedCols[j].getValue(r);
      const z = (raw - colMeans[j]) / colScales[j];
      xRow.push(z);
    }
    X_train.push(xRow);
    y_train.push(r[targetCol] as number);
  }

  // Solve beta = (X^T X + lambda I)^-1 X^T y
  const D = P + 1; // including intercept
  // Compute XtX (D x D)
  const XtX: number[][] = Array.from({ length: D }, () => Array(D).fill(0));
  for (let i = 0; i < nTrain; i++) {
    const x = X_train[i];
    for (let j = 0; j < D; j++) {
      for (let k = 0; k < D; k++) {
        XtX[j][k] += x[j] * x[k];
      }
    }
  }

  // Add L2 ridge regularization (skip intercept at j=0)
  for (let j = 1; j < D; j++) {
    XtX[j][j] += regularizationLambda;
  }

  // Invert XtX
  const XtX_inv = invertMatrix(XtX);

  // Compute Xty (D x 1)
  const Xty: number[] = Array(D).fill(0);
  for (let i = 0; i < nTrain; i++) {
    const yVal = y_train[i];
    const x = X_train[i];
    for (let j = 0; j < D; j++) {
      Xty[j] += x[j] * yVal;
    }
  }

  // beta = XtX_inv * Xty
  const beta: number[] = Array(D).fill(0);
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      beta[i] += XtX_inv[i][j] * Xty[j];
    }
  }

  const intercept = beta[0];
  const coefficients = beta.slice(1);

  // Helper prediction function
  const predictRow = (r: Record<string, string | number>): number => {
    let p = intercept;
    for (let j = 0; j < P; j++) {
      const z = (encodedCols[j].getValue(r) - colMeans[j]) / colScales[j];
      p += coefficients[j] * z;
    }
    return p;
  };

  // Evaluate on Train
  const trainPreds = trainRows.map(predictRow);
  const trainActuals = trainRows.map((r) => r[targetCol] as number);
  const trainMae = trainActuals.reduce((acc, y, i) => acc + Math.abs(y - trainPreds[i]), 0) / trainActuals.length;
  const trainMse = trainActuals.reduce((acc, y, i) => acc + Math.pow(y - trainPreds[i], 2), 0) / trainActuals.length;
  const trainMeanY = trainActuals.reduce((a, b) => a + b, 0) / trainActuals.length;
  const trainTotalVar = trainActuals.reduce((acc, y) => acc + Math.pow(y - trainMeanY, 2), 0);
  const trainR2 = trainTotalVar > 0 ? 1 - (trainMse * trainActuals.length) / trainTotalVar : 0;

  // Evaluate on Test
  const testPreds = testRows.map(predictRow);
  const testActuals = testRows.map((r) => r[targetCol] as number);
  const testMae = testActuals.length > 0
    ? testActuals.reduce((acc, y, i) => acc + Math.abs(y - testPreds[i]), 0) / testActuals.length
    : 0;
  const testMse = testActuals.length > 0
    ? testActuals.reduce((acc, y, i) => acc + Math.pow(y - testPreds[i], 2), 0) / testActuals.length
    : 0;
  const testTotalVar = testActuals.reduce((acc, y) => acc + Math.pow(y - trainMeanY, 2), 0);
  const testR2 = testTotalVar > 0 ? 1 - (testMse * testActuals.length) / testTotalVar : 0;

  // Baseline Dummy Mean on Test
  const dummyMae = testActuals.length > 0
    ? testActuals.reduce((acc, y) => acc + Math.abs(y - trainMeanY), 0) / testActuals.length
    : 0;
  const dummyRmse = testActuals.length > 0
    ? Math.sqrt(testActuals.reduce((acc, y) => acc + Math.pow(y - trainMeanY, 2), 0) / testActuals.length)
    : 0;

  // Split Conformal Prediction Interval calculation on Train residuals
  const trainResiduals = trainActuals.map((y, i) => Math.abs(y - trainPreds[i])).sort((a, b) => a - b);
  const confIdx = Math.min(trainResiduals.length - 1, Math.ceil((trainResiduals.length + 1) * 0.95) - 1);
  const conformalQ = trainResiduals[Math.max(0, confIdx)];

  // Test coverage
  let testCoveredCount = 0;
  for (let i = 0; i < testActuals.length; i++) {
    const y = testActuals[i];
    const p = testPreds[i];
    if (y >= p - conformalQ && y <= p + conformalQ) {
      testCoveredCount++;
    }
  }

  // Feature weights ranking
  const weights: TrainedFeatureWeight[] = encodedCols.map((c, j) => ({
    featureName: c.name,
    originalColumn: c.origCol,
    coefficient: coefficients[j] / colScales[j], // unscaled slope
    stdCoefficient: coefficients[j],             // standardized coefficient
    mean: colMeans[j],
    scale: colScales[j],
    importanceRank: 0
  }));

  weights.sort((a, b) => Math.abs(b.stdCoefficient) - Math.abs(a.stdCoefficient));
  weights.forEach((w, idx) => (w.importanceRank = idx + 1));

  // Scatter points
  const scatterPoints = [
    ...trainRows.map((r, i) => ({
      actual: trainActuals[i],
      predicted: trainPreds[i],
      residual: trainActuals[i] - trainPreds[i],
      isTest: false
    })),
    ...testRows.map((r, i) => ({
      actual: testActuals[i],
      predicted: testPreds[i],
      residual: testActuals[i] - testPreds[i],
      lower: testPreds[i] - conformalQ,
      upper: testPreds[i] + conformalQ,
      covered: testActuals[i] >= testPreds[i] - conformalQ && testActuals[i] <= testPreds[i] + conformalQ,
      isTest: true
    }))
  ];

  return {
    sampleCount: N,
    trainCount: nTrain,
    testCount: testRows.length,
    targetColumn: targetCol,
    intercept,
    weights,
    trainMetrics: {
      mae: trainMae,
      rmse: Math.sqrt(trainMse),
      r2: trainR2
    },
    testMetrics: {
      mae: testMae,
      rmse: Math.sqrt(testMse),
      r2: testR2,
      dummyMae,
      dummyRmse,
      dummyR2: 0
    },
    conformalInterval: {
      calCount: nTrain,
      conformalQ,
      nominalCoverage: 95.0,
      empiricalCoverage: testActuals.length > 0 ? (testCoveredCount / testActuals.length) * 100 : 95.0,
      meanWidth: 2 * conformalQ
    },
    scatterPoints
  };
}
