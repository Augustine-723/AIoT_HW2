import { parseCSV, inspectDataset, trainCustomLinearModel } from "../lib/custom-dataset-solver";

const sampleCSV = `R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.53,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
66051.52,182645.56,118148.2,Florida,103282.38
67532.53,105751.03,304768.73,Florida,107978.19
`;

const rows = parseCSV(sampleCSV);
console.log("Parsed rows:", rows.length);
const summary = inspectDataset(rows);
console.log("Columns:", summary.columns.map((c) => `${c.name} (${c.type})`));
const evalRes = trainCustomLinearModel(rows, "Profit", ["R&D Spend", "Administration", "Marketing Spend", "State"]);
console.log("Model Trained Successfully!");
console.log("Train RMSE:", evalRes.trainMetrics.rmse.toFixed(2));
console.log("Train R2:", evalRes.trainMetrics.r2.toFixed(4));
console.log("Test RMSE:", evalRes.testMetrics.rmse.toFixed(2));
console.log("Test R2:", evalRes.testMetrics.r2.toFixed(4));
console.log("Top features by weight:", evalRes.weights.map((w) => `${w.featureName}: ${w.stdCoefficient.toFixed(2)}`));
console.log("Conformal Interval q:", evalRes.conformalInterval?.conformalQ.toFixed(2));
