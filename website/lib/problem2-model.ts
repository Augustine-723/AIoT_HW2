/**
 * Authoritative Problem 2 (Appliances Energy Prediction) Inference Engine & Input Validation
 * Experiment ID: AIoT_HW2_P2_EXP01
 * Primary Model: M3 Ridge Regression (alpha=1000.0) & M2 RFE + OLS (k=6)
 * 
 * Model specification dynamically loaded from exported problem2/model_spec.json
 * and problem2/m2_model_spec.json to ensure single-source-of-truth and floating-point parity.
 */

import modelSpecRaw from "../data/problem2/model_spec.json";
import m2SpecRaw from "../data/problem2/m2_model_spec.json";

export interface AppliancesInputFeatures {
  // Lighting (1)
  lights: number;       // Energy use of light fixtures in Wh (e.g. 0 - 60)
  
  // Indoor Temperature in Celsius (9)
  T1: number;           // Kitchen area temp
  T2: number;           // Living room area temp
  T3: number;           // Laundry room area temp
  T4: number;           // Office room temp
  T5: number;           // Bathroom temp
  T6: number;           // Outside building (north side) temp
  T7: number;           // Ironing room temp
  T8: number;           // Teenager room 2 temp
  T9: number;           // Parents room temp
  
  // Indoor Relative Humidity in % (9)
  RH_1: number;         // Kitchen humidity
  RH_2: number;         // Living room humidity
  RH_3: number;         // Laundry room humidity
  RH_4: number;         // Office room humidity
  RH_5: number;         // Bathroom humidity
  RH_6: number;         // Outside building north side humidity
  RH_7: number;         // Ironing room humidity
  RH_8: number;         // Teenager room 2 humidity
  RH_9: number;         // Parents room humidity
  
  // Outdoor Weather from Chièvres Airport (6)
  T_out: number;        // Outdoor temp in Celsius
  Press_mm_hg: number;  // Atmospheric pressure in mm Hg
  RH_out: number;       // Outdoor relative humidity in %
  Windspeed: number;    // Wind speed in m/s
  Visibility: number;   // Visibility in km
  Tdewpoint: number;    // Dewpoint temp in Celsius
  
  // Time Schedule (1 primary input -> derives sin_hour, cos_hour, is_weekend)
  hour: number;         // Hour of day (0 - 23)
  is_weekend?: number;  // Weekend indicator (0: Weekday, 1: Weekend)
}

export interface ValidationError {
  field: keyof AppliancesInputFeatures;
  message: string;
}

export interface ValidationWarning {
  field: string;
  title: string;
  message: string;
  severity: "low" | "medium" | "high";
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  sanitized: AppliancesInputFeatures;
}

export interface FeatureContribution {
  feature_name: string;
  category: "Lighting" | "Indoor Temperature" | "Indoor Humidity" | "Outdoor Weather" | "Time Schedule";
  raw_value: number;
  standardized_value: number;
  coefficient: number;
  contribution: number;
  description: string;
}

export interface AppliancesPredictionResult {
  modelId: string;
  modelName: string;
  rawPointPrediction: number;
  clippedPointPrediction: number;
  conformalQuantile: number;
  piLowerRaw: number;
  piUpperRaw: number;
  piLowerClipped: number;
  piUpperClipped: number;
  nominalCoveragePct: number;
  empiricalTestCoveragePct: number;
  contributions: FeatureContribution[];
  interceptContribution: number;
  sanitizedInputs: AppliancesInputFeatures;
}

export const PROBLEM2_M3_SPEC = modelSpecRaw;
export const PROBLEM2_M2_SPEC = m2SpecRaw;

// Group definitions for 26 raw inputs
export const FEATURE_GROUPS = {
  lighting: {
    id: "lighting",
    title: "Lighting (燈光能耗)",
    enTitle: "Lighting Telemetry",
    features: ["lights"] as (keyof AppliancesInputFeatures)[]
  },
  indoorTemp: {
    id: "indoorTemp",
    title: "Indoor Temperatures (室內各區溫度 °C)",
    enTitle: "Room Temperatures",
    features: ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"] as (keyof AppliancesInputFeatures)[]
  },
  indoorHum: {
    id: "indoorHum",
    title: "Indoor Humidities (室內各區相對濕度 %)",
    enTitle: "Room Humidities",
    features: ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"] as (keyof AppliancesInputFeatures)[]
  },
  outdoorWeather: {
    id: "outdoorWeather",
    title: "Outdoor Weather (氣象站大氣觀測)",
    enTitle: "Chièvres Weather Station",
    features: ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"] as (keyof AppliancesInputFeatures)[]
  },
  timeSchedule: {
    id: "timeSchedule",
    title: "Time Schedule (時段與作息)",
    enTitle: "Daily Time Schedule",
    features: ["hour"] as (keyof AppliancesInputFeatures)[]
  }
};

export const FEATURE_METADATA: Record<string, { label: string; unit: string; min: number; max: number; default: number; desc: string }> = {
  lights: { label: "客廳與廚房燈具能耗", unit: "Wh", min: 0, max: 70, default: 0, desc: "室內所有燈具之即時瓦時耗電量" },
  T1: { label: "廚房溫度 (Kitchen)", unit: "°C", min: 16.0, max: 27.0, default: 21.1, desc: "廚房區域 ZigBee 感測節點" },
  RH_1: { label: "廚房濕度 (Kitchen)", unit: "%", min: 25.0, max: 65.0, default: 40.4, desc: "廚房區域相對濕度" },
  T2: { label: "客廳溫度 (Living Room)", unit: "°C", min: 16.0, max: 30.0, default: 19.6, desc: "客廳區域溫度感測" },
  RH_2: { label: "客廳濕度 (Living Room)", unit: "%", min: 20.0, max: 60.0, default: 40.9, desc: "客廳區域相對濕度" },
  T3: { label: "洗衣間溫度 (Laundry)", unit: "°C", min: 17.0, max: 30.0, default: 21.4, desc: "洗衣間區域溫度" },
  RH_3: { label: "洗衣間濕度 (Laundry)", unit: "%", min: 28.0, max: 50.0, default: 39.4, desc: "洗衣間區域相對濕度" },
  T4: { label: "辦公室溫度 (Office)", unit: "°C", min: 15.0, max: 26.0, default: 20.9, desc: "居家辦公室區域溫度" },
  RH_4: { label: "辦公室濕度 (Office)", unit: "%", min: 25.0, max: 52.0, default: 39.0, desc: "辦公室相對濕度" },
  T5: { label: "浴室溫度 (Bathroom)", unit: "°C", min: 15.0, max: 26.0, default: 19.6, desc: "浴室區域溫度" },
  RH_5: { label: "浴室濕度 (Bathroom)", unit: "%", min: 28.0, max: 95.0, default: 51.0, desc: "浴室相對濕度" },
  T6: { label: "北側外牆溫度 (North Wall)", unit: "°C", min: -6.0, max: 28.0, default: 7.9, desc: "建築北側建築外部溫度" },
  RH_6: { label: "北側外牆濕度 (North Wall)", unit: "%", min: 1.0, max: 100.0, default: 54.6, desc: "建築北側外部相對濕度" },
  T7: { label: "燙衣間溫度 (Ironing)", unit: "°C", min: 15.0, max: 26.0, default: 20.3, desc: "燙衣間區域溫度" },
  RH_7: { label: "燙衣間濕度 (Ironing)", unit: "%", min: 23.0, max: 55.0, default: 35.3, desc: "燙衣間相對濕度" },
  T8: { label: "青少年臥室溫度 (Teenager)", unit: "°C", min: 16.0, max: 28.0, default: 22.2, desc: "青少年房 2 溫度" },
  RH_8: { label: "青少年臥室濕度 (Teenager)", unit: "%", min: 29.0, max: 60.0, default: 42.9, desc: "青少年房 2 相對濕度" },
  T9: { label: "主臥室溫度 (Parents)", unit: "°C", min: 14.0, max: 25.0, default: 19.4, desc: "父母主臥室區域溫度" },
  RH_9: { label: "主臥室濕度 (Parents)", unit: "%", min: 29.0, max: 55.0, default: 41.5, desc: "主臥室相對濕度" },
  T_out: { label: "氣象站室外溫度", unit: "°C", min: -5.0, max: 27.0, default: 7.4, desc: "Chièvres 機場氣象站溫度" },
  Press_mm_hg: { label: "大氣氣壓", unit: "mm Hg", min: 729.0, max: 773.0, default: 755.5, desc: "氣象站氣壓計讀數" },
  RH_out: { label: "氣象站室外濕度", unit: "%", min: 24.0, max: 100.0, default: 79.2, desc: "氣象站戶外相對濕度" },
  Windspeed: { label: "戶外風速", unit: "m/s", min: 0.0, max: 14.0, default: 4.0, desc: "氣象站觀測風速" },
  Visibility: { label: "大氣能見度", unit: "km", min: 1.0, max: 66.0, default: 40.0, desc: "氣象站觀測水平能見度" },
  Tdewpoint: { label: "露點溫度", unit: "°C", min: -7.0, max: 16.0, default: 3.7, desc: "大氣露點溫度" },
  hour: { label: "預測時段 (Hour)", unit: "時", min: 0, max: 23, default: 17, desc: "當前一天之 24 小時時段 (0 - 23)" }
};

export const DEFAULT_APPLIANCES_INPUTS: AppliancesInputFeatures = {
  lights: 0,
  T1: 21.1,
  RH_1: 40.4,
  T2: 19.6,
  RH_2: 40.9,
  T3: 21.4,
  RH_3: 39.4,
  T4: 20.9,
  RH_4: 39.0,
  T5: 19.6,
  RH_5: 51.0,
  T6: 7.9,
  RH_6: 54.6,
  T7: 20.3,
  RH_7: 35.3,
  T8: 22.2,
  RH_8: 42.9,
  T9: 19.4,
  RH_9: 41.5,
  T_out: 7.4,
  Press_mm_hg: 755.5,
  RH_out: 79.2,
  Windspeed: 4.0,
  Visibility: 40.0,
  Tdewpoint: 3.7,
  hour: 17,
  is_weekend: 0
};

export const APPLIANCES_PRESETS: { name: string; desc: string; values: Partial<AppliancesInputFeatures> }[] = [
  {
    name: "晨間甦醒與早餐 (Morning Wakeup)",
    desc: "早晨 07:00，廚房與客廳活動開始，室內溫度微升，燈具開啟 20 Wh",
    values: { hour: 7, lights: 20, T1: 20.8, T2: 19.2, RH_1: 42.5, is_weekend: 0 }
  },
  {
    name: "陽光正午無人期 (Sunny Midday)",
    desc: "中午 12:00 工作日，住戶大多外出，燈具關閉 (0 Wh)，室外氣溫上升",
    values: { hour: 12, lights: 0, T_out: 14.5, T6: 15.2, RH_out: 55.0, is_weekend: 0 }
  },
  {
    name: "傍晚烹飪與休閒尖峰 (Evening Peak)",
    desc: "晚間 18:00，住戶返家烹飪與洗衣，燈具全開 (40 Wh)，廚房濕度升高",
    values: { hour: 18, lights: 40, T1: 22.5, RH_1: 46.0, T3: 22.0, RH_3: 44.0, is_weekend: 0 }
  },
  {
    name: "週末午後聚會 (Weekend Afternoon)",
    desc: "週六 15:00，客廳與燙衣間多人活動，室內濕度舒適，燈具微開",
    values: { hour: 15, lights: 10, T2: 21.5, T7: 21.0, RH_2: 43.0, is_weekend: 1 }
  },
  {
    name: "深夜沉睡靜止 (Night Sleep)",
    desc: "深夜 03:00，所有住戶入睡，燈具關閉，僅維持基本待機能耗",
    values: { hour: 3, lights: 0, T9: 18.8, T8: 20.5, T_out: 4.0, RH_out: 88.0, is_weekend: 0 }
  }
];

export function validateAppliancesInputs(inputs: Partial<AppliancesInputFeatures>): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const sanitized = { ...DEFAULT_APPLIANCES_INPUTS, ...inputs };

  // Sanitize and clamp numeric features
  for (const [key, meta] of Object.entries(FEATURE_METADATA)) {
    const k = key as keyof AppliancesInputFeatures;
    let val = sanitized[k];
    if (typeof val !== "number" || isNaN(val)) {
      errors.push({ field: k, message: `${meta.label} 必須為合法數值。` });
      sanitized[k] = meta.default as never;
    } else {
      if (val < meta.min || val > meta.max) {
        warnings.push({
          field: key,
          title: "數值超出典型歷史邊界",
          message: `${meta.label} 數值 ${val} 超出歷史常見範圍 [${meta.min}, ${meta.max}]，推論將以模型最佳泛化點鉗制或外推計算。`,
          severity: "low"
        });
        val = Math.max(meta.min, Math.min(meta.max, val));
        sanitized[k] = val as never;
      }
    }
  }

  // Cross-variable domain sanity warnings
  if (sanitized.lights > 50 && (sanitized.hour >= 0 && sanitized.hour <= 5)) {
    warnings.push({
      field: "lights",
      title: "深夜高燈光耗能",
      message: `當前時段為深夜 ${sanitized.hour}:00，但燈光能耗達 ${sanitized.lights} Wh，可能屬於罕見作息或派對情境。`,
      severity: "medium"
    });
  }

  if (Math.abs(sanitized.T_out - sanitized.T6) > 15) {
    warnings.push({
      field: "T6",
      title: "外牆與氣象站溫度懸殊",
      message: `建築北側外牆溫度 (${sanitized.T6}°C) 與氣象站觀測 (${sanitized.T_out}°C) 溫差超過 15°C，可能受局部微氣候或強烈日照影響。`,
      severity: "medium"
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    sanitized
  };
}

export function predictAppliancesEnergy(
  inputs: AppliancesInputFeatures,
  modelType: "M3_RIDGE" | "M2_RFE" = "M3_RIDGE"
): AppliancesPredictionResult {
  const validation = validateAppliancesInputs(inputs);
  const clean = validation.sanitized;

  const spec = modelType === "M3_RIDGE" ? PROBLEM2_M3_SPEC : PROBLEM2_M2_SPEC;
  const intercept = spec.intercept;
  const calQ = spec.calibration_q;
  const nominalCoverage = spec.nominal_coverage_pct;
  const empiricalCoverage = spec.empirical_test_coverage_pct;

  // Compute derived time features
  const sinHour = Math.sin((2 * Math.PI * clean.hour) / 24.0);
  const cosHour = Math.cos((2 * Math.PI * clean.hour) / 24.0);
  const isWeekend = clean.is_weekend ?? 0;

  // Complete feature mapping dictionary
  const featureValues: Record<string, number> = {
    lights: clean.lights,
    T1: clean.T1,
    RH_1: clean.RH_1,
    T2: clean.T2,
    RH_2: clean.RH_2,
    T3: clean.T3,
    RH_3: clean.RH_3,
    T4: clean.T4,
    RH_4: clean.RH_4,
    T5: clean.T5,
    RH_5: clean.RH_5,
    T6: clean.T6,
    RH_6: clean.RH_6,
    T7: clean.T7,
    RH_7: clean.RH_7,
    T8: clean.T8,
    RH_8: clean.RH_8,
    T9: clean.T9,
    RH_9: clean.RH_9,
    T_out: clean.T_out,
    Press_mm_hg: clean.Press_mm_hg,
    RH_out: clean.RH_out,
    Windspeed: clean.Windspeed,
    Visibility: clean.Visibility,
    Tdewpoint: clean.Tdewpoint,
    hour: clean.hour,
    sin_hour: sinHour,
    cos_hour: cosHour,
    is_weekend: isWeekend
  };

  let rawPred = intercept;
  const contributions: FeatureContribution[] = [];

  for (const fSpec of spec.feature_specs) {
    const rawVal = featureValues[fSpec.feature] ?? 0;
    const stdVal = (rawVal - fSpec.mean) / fSpec.scale;
    const contrib = fSpec.coefficient * stdVal;
    rawPred += contrib;

    let category: FeatureContribution["category"] = "Indoor Temperature";
    if (fSpec.feature === "lights") category = "Lighting";
    else if (fSpec.feature.startsWith("RH_") && fSpec.feature !== "RH_out") category = "Indoor Humidity";
    else if (["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"].includes(fSpec.feature)) category = "Outdoor Weather";
    else if (["hour", "sin_hour", "cos_hour", "is_weekend"].includes(fSpec.feature)) category = "Time Schedule";

    const meta = FEATURE_METADATA[fSpec.feature];
    const desc = meta ? `${meta.label} (${rawVal} ${meta.unit})` : fSpec.feature;

    contributions.push({
      feature_name: fSpec.feature,
      category,
      raw_value: rawVal,
      standardized_value: stdVal,
      coefficient: fSpec.coefficient,
      contribution: contrib,
      description: desc
    });
  }

  const clippedPred = Math.max(0.0, rawPred);
  const piLowerRaw = rawPred - calQ;
  const piUpperRaw = rawPred + calQ;
  const piLowerClipped = Math.max(0.0, piLowerRaw);
  const piUpperClipped = piUpperRaw;

  // Sort contributions by absolute impact
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const specModelId = ("model_id" in spec ? (spec as { model_id: string }).model_id : (spec as { best_model_id: string }).best_model_id) ?? (modelType === "M3_RIDGE" ? "M3" : "M2");

  return {
    modelId: specModelId,
    modelName: spec.model_name,
    rawPointPrediction: rawPred,
    clippedPointPrediction: clippedPred,
    conformalQuantile: calQ,
    piLowerRaw,
    piUpperRaw,
    piLowerClipped,
    piUpperClipped,
    nominalCoveragePct: nominalCoverage,
    empiricalTestCoveragePct: empiricalCoverage,
    contributions,
    interceptContribution: intercept,
    sanitizedInputs: clean
  };
}
