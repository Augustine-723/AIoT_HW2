/**
 * Authoritative M2 (RFE + OLS) Inference Engine & Input Validation
 * Experiment ID: AIoT_HW2_PHASE3_EXP01
 * Model ID: M2_RFE_OLS
 * 
 * Model specification dynamically loaded from exported m2_model_spec.json
 * to eliminate manually duplicated parameters and ensure single-source-of-truth.
 */

import m2ModelSpecRaw from "../data/m2_model_spec.json";

export interface BikeInputFeatures {
  yr: number;         // 0: 2011, 1: 2012
  mnth: number;       // 1-12
  hr: number;         // 0-23
  weekday: number;    // 0-6 (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
  holiday: number;    // 0 or 1
  workingday: number; // 0 or 1
  season: number;     // 1: Spring, 2: Summer, 3: Fall, 4: Winter
  weathersit: number; // 1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow
  temp: number;       // Normalized temperature in [0, 1] (t_real / 41)
  atemp: number;      // Normalized feeling temperature in [0, 1] (at_real / 50)
  hum: number;        // Normalized humidity in [0, 1] (hum_real / 100)
  windspeed: number;  // Normalized wind speed in [0, 1] (wind_real / 67)
}

export interface ValidationError {
  field: keyof BikeInputFeatures;
  message: string;
}

export interface ValidationWarning {
  code: "WEEKEND_WORKINGDAY" | "HOLIDAY_WORKINGDAY" | "WEEKDAY_NONWORKING" | "MONTH_SEASON_MISMATCH" | "TEMP_ATEMP_DIVERGENCE" | "YEAR_EXTRAPOLATION";
  title: string;
  message: string;
  severity: "low" | "medium" | "high";
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  sanitized: BikeInputFeatures;
}

export interface FeatureContribution {
  feature_name: string;
  category: "Base Intercept" | "Season" | "Month" | "Hour" | "Weather" | "Climate Numerics" | "Year Trend";
  value: number;
  coefficient: number;
  contribution: number;
  description: string;
}

export interface PredictionResult {
  rawPointPrediction: number;
  clippedPointPrediction: number;
  conformalQuantile: number;
  piLowerRaw: number;
  piUpperRaw: number;
  piLowerClipped: number;
  piUpperClipped: number;
  intercept: number;
  featureContributions: FeatureContribution[];
  experimentId: string;
  modelId: string;
  validation: ValidationResult;
}

interface M2RawSpec {
  intercept: number;
  conformal_quantile: number;
  coefficients: Record<string, number>;
}

const rawSpec = m2ModelSpecRaw as unknown as M2RawSpec;

export const M2_SPEC = {
  experimentId: "AIoT_HW2_PHASE3_EXP01",
  modelId: "M2_RFE_OLS",
  intercept: rawSpec.intercept,
  conformalQuantile: rawSpec.conformal_quantile,
  coefficients: rawSpec.coefficients,
};

/**
 * Centralized Input Validation & Consistency Checking
 * Validates all 12 original predictors against their allowed domains,
 * and detects inconsistent calendar or physical climate combinations.
 */
export function validateBikeInputs(input: Partial<BikeInputFeatures>): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  // Helper clamp
  const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

  // 1. yr
  let yr = typeof input.yr === "number" ? Math.round(input.yr) : 1;
  if (yr !== 0 && yr !== 1) {
    errors.push({ field: "yr", message: "年份指標 (yr) 必須為 0 (2011) 或 1 (2012)" });
    warnings.push({
      code: "YEAR_EXTRAPOLATION",
      title: "不可外推警告",
      message: "本模型基於 2011–2012 歷史資料擬合，不可外推至未來年份。",
      severity: "high",
    });
    yr = clamp(yr, 0, 1);
  }

  // 2. mnth
  let mnth = typeof input.mnth === "number" ? Math.round(input.mnth) : 10;
  if (mnth < 1 || mnth > 12) {
    errors.push({ field: "mnth", message: "月份 (mnth) 必須介於 1 到 12 之間" });
    mnth = clamp(mnth, 1, 12);
  }

  // 3. hr
  let hr = typeof input.hr === "number" ? Math.round(input.hr) : 8;
  if (hr < 0 || hr > 23) {
    errors.push({ field: "hr", message: "小時時段 (hr) 必須介於 0 到 23 之間" });
    hr = clamp(hr, 0, 23);
  }

  // 4. weekday
  let weekday = typeof input.weekday === "number" ? Math.round(input.weekday) : 2;
  if (weekday < 0 || weekday > 6) {
    errors.push({ field: "weekday", message: "星期 (weekday) 必須介於 0 (週日) 到 6 (週六) 之間" });
    weekday = clamp(weekday, 0, 6);
  }

  // 5. holiday
  let holiday = typeof input.holiday === "number" ? Math.round(input.holiday) : 0;
  if (holiday !== 0 && holiday !== 1) {
    errors.push({ field: "holiday", message: "國定假日 (holiday) 必須為 0 (否) 或 1 (是)" });
    holiday = clamp(holiday, 0, 1);
  }

  // 6. workingday
  let workingday = typeof input.workingday === "number" ? Math.round(input.workingday) : 1;
  if (workingday !== 0 && workingday !== 1) {
    errors.push({ field: "workingday", message: "工作日 (workingday) 必須為 0 (否) 或 1 (是)" });
    workingday = clamp(workingday, 0, 1);
  }

  // 7. season
  let season = typeof input.season === "number" ? Math.round(input.season) : 4;
  if (season < 1 || season > 4) {
    errors.push({ field: "season", message: "季節 (season) 必須為 1(春), 2(夏), 3(秋), 4(冬)" });
    season = clamp(season, 1, 4);
  }

  // 8. weathersit
  let weathersit = typeof input.weathersit === "number" ? Math.round(input.weathersit) : 1;
  if (weathersit < 1 || weathersit > 4) {
    errors.push({ field: "weathersit", message: "天候代碼 (weathersit) 必須介於 1 到 4 之間" });
    weathersit = clamp(weathersit, 1, 4);
  }

  // 9. temp
  let temp = typeof input.temp === "number" ? input.temp : 0.38;
  if (isNaN(temp) || temp < 0 || temp > 1) {
    errors.push({ field: "temp", message: "正規化氣溫 (temp) 必須介於 0.00 到 1.00 之間" });
    temp = clamp(isNaN(temp) ? 0.38 : temp, 0, 1);
  }

  // 10. atemp
  let atemp = typeof input.atemp === "number" ? input.atemp : 0.39;
  if (isNaN(atemp) || atemp < 0 || atemp > 1) {
    errors.push({ field: "atemp", message: "正規化體感溫度 (atemp) 必須介於 0.00 到 1.00 之間" });
    atemp = clamp(isNaN(atemp) ? 0.39 : atemp, 0, 1);
  }

  // 11. hum
  let hum = typeof input.hum === "number" ? input.hum : 0.50;
  if (isNaN(hum) || hum < 0 || hum > 1) {
    errors.push({ field: "hum", message: "正規化濕度 (hum) 必須介於 0.00 到 1.00 之間" });
    hum = clamp(isNaN(hum) ? 0.50 : hum, 0, 1);
  }

  // 12. windspeed
  let windspeed = typeof input.windspeed === "number" ? input.windspeed : 0.22;
  if (isNaN(windspeed) || windspeed < 0 || windspeed > 1) {
    errors.push({ field: "windspeed", message: "正規化風速 (windspeed) 必須介於 0.00 到 1.00 之間" });
    windspeed = clamp(isNaN(windspeed) ? 0.22 : windspeed, 0, 1);
  }

  // Calendar Inconsistency Warnings (UCI Dataset Domain Logic)
  // Warning A: Weekend marked as working day
  if ((weekday === 0 || weekday === 6) && workingday === 1) {
    warnings.push({
      code: "WEEKEND_WORKINGDAY",
      title: "曆法組合警示：週末標記為工作日",
      message: `所選星期為 ${weekday === 0 ? "週日 (0)" : "週六 (6)"}，但工作日標記為 1。UCI 資料集定義工作日為「既非週末亦非假日」，實務上週末通常 workingday = 0。`,
      severity: "medium",
    });
  }

  // Warning B: Holiday marked as working day
  if (holiday === 1 && workingday === 1) {
    warnings.push({
      code: "HOLIDAY_WORKINGDAY",
      title: "曆法組合警示：國定假日標記為工作日",
      message: "所選為國定假日 (holiday=1)，但工作日標記為 1。依資料集定義，公眾假期應為非工作日 (workingday=0)。",
      severity: "high",
    });
  }

  // Warning C: Regular weekday marked as non-working day without holiday
  if (weekday >= 1 && weekday <= 5 && holiday === 0 && workingday === 0) {
    warnings.push({
      code: "WEEKDAY_NONWORKING",
      title: "曆法提示：平日標記為非工作日",
      message: "所選為週一至週五平日且非國定假日，但工作日標記為 0。可能代表特殊停班日或調休。",
      severity: "low",
    });
  }

  // Warning D: Month and Season apparent mismatch
  // Dataset definitions: Season 1: Spring (~Jan-Mar), Season 2: Summer (~Apr-Jun), Season 3: Fall (~Jul-Sep), Season 4: Winter (~Oct-Dec)
  const typicalSeasonForMonth = (m: number) => {
    if (m >= 1 && m <= 3) return 1;
    if (m >= 4 && m <= 6) return 2;
    if (m >= 7 && m <= 9) return 3;
    return 4;
  };
  const expectedSeason = typicalSeasonForMonth(mnth);
  if (Math.abs(expectedSeason - season) >= 2) {
    const seasonNames = ["", "春季", "夏季", "秋季", "冬季"];
    warnings.push({
      code: "MONTH_SEASON_MISMATCH",
      title: "曆法提示：月份與季節相差較大",
      message: `所選月份為 ${mnth} 月，但季節指定為 ${seasonNames[season]}。在北半球溫帶氣候通常不屬於此季節分期。`,
      severity: "low",
    });
  }

  // Warning E: Large Temp vs Atemp Divergence
  if (Math.abs(temp - atemp) > 0.30) {
    warnings.push({
      code: "TEMP_ATEMP_DIVERGENCE",
      title: "氣象物理異常提示：氣溫與體感溫度相差過大",
      message: `正規化氣溫 (${temp.toFixed(2)}) 與體感溫度 (${atemp.toFixed(2)}) 差異超過 0.30（換算實際溫差逾 15°C），在真實戶外微氣候極為罕見。`,
      severity: "medium",
    });
  }

  const sanitized: BikeInputFeatures = {
    yr,
    mnth,
    hr,
    weekday,
    holiday,
    workingday,
    season,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed,
  };

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    sanitized,
  };
}

/**
 * Executes authoritative M2 inference for given 12 input features.
 * All model parameters read directly from authoritative M2_SPEC.
 */
export function predictBikeDemand(rawInput: Partial<BikeInputFeatures>): PredictionResult {
  const validation = validateBikeInputs(rawInput);
  const input = validation.sanitized;

  const coefs = M2_SPEC.coefficients;
  let rawPred = M2_SPEC.intercept;
  const contributions: FeatureContribution[] = [];

  // Base intercept contribution
  contributions.push({
    feature_name: "Intercept (截距常數)",
    category: "Base Intercept",
    value: 1,
    coefficient: M2_SPEC.intercept,
    contribution: M2_SPEC.intercept,
    description: "基礎線性迴歸基準常數",
  });

  // 1. Season (reference = 1: Spring)
  if (input.season === 2 && coefs.cat__season_2 !== undefined) {
    const val = coefs.cat__season_2;
    rawPred += val;
    contributions.push({
      feature_name: "Season = Summer (夏季)",
      category: "Season",
      value: 1,
      coefficient: coefs.cat__season_2,
      contribution: val,
      description: "夏季租借意願相較春季提升",
    });
  } else if (input.season === 3 && coefs.cat__season_3 !== undefined) {
    const val = coefs.cat__season_3;
    rawPred += val;
    contributions.push({
      feature_name: "Season = Fall (秋季)",
      category: "Season",
      value: 1,
      coefficient: coefs.cat__season_3,
      contribution: val,
      description: "秋季氣候宜人，租借需求顯著高於春季",
    });
  } else if (input.season === 4 && coefs.cat__season_4 !== undefined) {
    const val = coefs.cat__season_4;
    rawPred += val;
    contributions.push({
      feature_name: "Season = Winter (冬季)",
      category: "Season",
      value: 1,
      coefficient: coefs.cat__season_4,
      contribution: val,
      description: "冬季調整係數（經溫度控制後）",
    });
  }

  // 2. Month (only months 7 and 8 retained by RFE)
  if (input.mnth === 7 && coefs.cat__mnth_7 !== undefined) {
    const val = coefs.cat__mnth_7;
    rawPred += val;
    contributions.push({
      feature_name: "Month = 7 (七月)",
      category: "Month",
      value: 1,
      coefficient: coefs.cat__mnth_7,
      contribution: val,
      description: "盛夏高溫抑制部分日間騎乘需求",
    });
  } else if (input.mnth === 8 && coefs.cat__mnth_8 !== undefined) {
    const val = coefs.cat__mnth_8;
    rawPred += val;
    contributions.push({
      feature_name: "Month = 8 (八月)",
      category: "Month",
      value: 1,
      coefficient: coefs.cat__mnth_8,
      contribution: val,
      description: "暑期高溫休假效應調整",
    });
  }

  // 3. Hour (hours 6 through 23 retained by RFE, hours 0-5 are reference/eliminated)
  const hrKey = `cat__hr_${input.hr}`;
  if (coefs[hrKey] !== undefined) {
    const val = coefs[hrKey];
    rawPred += val;
    let desc = `時段效應 (${input.hr}:00)`;
    if (input.hr === 8 || input.hr === 17 || input.hr === 18) {
      desc = `尖峰通勤潮時段 (${input.hr}:00)，係數最高`;
    }
    contributions.push({
      feature_name: `Hour = ${input.hr}:00`,
      category: "Hour",
      value: 1,
      coefficient: coefs[hrKey],
      contribution: val,
      description: desc,
    });
  }

  // 4. Weather Situation (reference = 1: Clear, 2: Mist eliminated by RFE)
  if (input.weathersit === 3 && coefs.cat__weathersit_3 !== undefined) {
    const val = coefs.cat__weathersit_3;
    rawPred += val;
    contributions.push({
      feature_name: "Weather = Light Rain/Snow (小雨/小雪)",
      category: "Weather",
      value: 1,
      coefficient: coefs.cat__weathersit_3,
      contribution: val,
      description: "降雨造成路面濕滑，顯著降低騎乘需求",
    });
  } else if (input.weathersit === 4 && coefs.cat__weathersit_4 !== undefined) {
    const val = coefs.cat__weathersit_4;
    rawPred += val;
    contributions.push({
      feature_name: "Weather = Heavy Rain/Snow (大豪雨/暴風雪)",
      category: "Weather",
      value: 1,
      coefficient: coefs.cat__weathersit_4,
      contribution: val,
      description: "極端惡劣天氣，強烈抑制戶外騎乘",
    });
  }

  // 5. Continuous Climate Numerics
  if (coefs.num__temp !== undefined) {
    const tempCont = input.temp * coefs.num__temp;
    rawPred += tempCont;
    contributions.push({
      feature_name: "Normalized Temperature (正規化氣溫)",
      category: "Climate Numerics",
      value: input.temp,
      coefficient: coefs.num__temp,
      contribution: tempCont,
      description: `氣溫 ${Math.round(input.temp * 41)}°C (標準化值 ${input.temp.toFixed(2)})，正向促進需求`,
    });
  }

  if (coefs.num__atemp !== undefined) {
    const atempCont = input.atemp * coefs.num__atemp;
    rawPred += atempCont;
    contributions.push({
      feature_name: "Normalized Feeling Temp (正規化體感溫度)",
      category: "Climate Numerics",
      value: input.atemp,
      coefficient: coefs.num__atemp,
      contribution: atempCont,
      description: `體感溫度 ${Math.round(input.atemp * 50)}°C (標準化值 ${input.atemp.toFixed(2)})，主導騎乘舒適度`,
    });
  }

  if (coefs.num__hum !== undefined) {
    const humCont = input.hum * coefs.num__hum;
    rawPred += humCont;
    contributions.push({
      feature_name: "Normalized Humidity (正規化濕度)",
      category: "Climate Numerics",
      value: input.hum,
      coefficient: coefs.num__hum,
      contribution: humCont,
      description: `濕度 ${Math.round(input.hum * 100)}% (標準化值 ${input.hum.toFixed(2)})，潮濕微悶負向抑制`,
    });
  }

  if (coefs.num__windspeed !== undefined) {
    const windCont = input.windspeed * coefs.num__windspeed;
    rawPred += windCont;
    contributions.push({
      feature_name: "Normalized Wind Speed (正規化風速)",
      category: "Climate Numerics",
      value: input.windspeed,
      coefficient: coefs.num__windspeed,
      contribution: windCont,
      description: `風速 ${Math.round(input.windspeed * 67)} km/h (標準化值 ${input.windspeed.toFixed(2)})，強風負向抑制`,
    });
  }

  // 6. Year Trend (0 = 2011, 1 = 2012)
  if (input.yr === 1 && coefs.num__yr !== undefined) {
    const val = coefs.num__yr;
    rawPred += val;
    contributions.push({
      feature_name: "Year = 2012 (系統營運成長趨勢)",
      category: "Year Trend",
      value: 1,
      coefficient: coefs.num__yr,
      contribution: val,
      description: "2012 年系統知名度與站點擴增帶來之整體基期成長 (+82.1 輛/時)",
    });
  }

  const q = M2_SPEC.conformalQuantile;
  const rawPoint = rawPred;
  const clippedPoint = Math.max(0, rawPred);
  const piLowerRaw = rawPoint - q;
  const piUpperRaw = rawPoint + q;
  const piLowerClipped = Math.max(0, piLowerRaw);
  const piUpperClipped = piUpperRaw;

  return {
    rawPointPrediction: rawPoint,
    clippedPointPrediction: clippedPoint,
    conformalQuantile: q,
    piLowerRaw,
    piUpperRaw,
    piLowerClipped,
    piUpperClipped,
    intercept: M2_SPEC.intercept,
    featureContributions: contributions,
    experimentId: M2_SPEC.experimentId,
    modelId: M2_SPEC.modelId,
    validation,
  };
}

export const PRESET_SCENARIOS: { name: string; tag: string; description: string; features: BikeInputFeatures }[] = [
  {
    name: "晨峰工作日通勤 (Morning Peak Commute)",
    tag: "尖峰通勤",
    description: "2012 年 10 月週二早上 08:00，晴天涼爽，為都市單車最高需求時段之一",
    features: {
      yr: 1,
      mnth: 10,
      hr: 8,
      weekday: 2,
      holiday: 0,
      workingday: 1,
      season: 4,
      weathersit: 1,
      temp: 0.38,
      atemp: 0.39,
      hum: 0.50,
      windspeed: 0.22,
    },
  },
  {
    name: "傍晚工作日下班峰 (Evening Peak Commute)",
    tag: "尖峰通勤",
    description: "2012 年 10 月週五傍晚 17:00，氣候溫和晴朗，下班及放學返家潮",
    features: {
      yr: 1,
      mnth: 10,
      hr: 17,
      weekday: 5,
      holiday: 0,
      workingday: 1,
      season: 4,
      weathersit: 1,
      temp: 0.52,
      atemp: 0.50,
      hum: 0.45,
      windspeed: 0.19,
    },
  },
  {
    name: "週末午後休閒騎乘 (Weekend Leisure)",
    tag: "休閒觀光",
    description: "2012 年 7 月週六下午 14:00，晴朗夏日，非通勤休閒出遊高峰",
    features: {
      yr: 1,
      mnth: 7,
      hr: 14,
      weekday: 6,
      holiday: 0,
      workingday: 0,
      season: 3,
      weathersit: 1,
      temp: 0.76,
      atemp: 0.71,
      hum: 0.48,
      windspeed: 0.16,
    },
  },
  {
    name: "低溫暴雨工作日 (Cold Rainy Day)",
    tag: "惡劣天候",
    description: "2012 年 11 月週三中午 12:00，遭遇降雨與強風，需求大幅受壓抑",
    features: {
      yr: 1,
      mnth: 11,
      hr: 12,
      weekday: 3,
      holiday: 0,
      workingday: 1,
      season: 4,
      weathersit: 3,
      temp: 0.28,
      atemp: 0.27,
      hum: 0.88,
      windspeed: 0.39,
    },
  },
  {
    name: "深夜冷門時段 (Deep Night Baseline)",
    tag: "深夜低需求",
    description: "2011 年 1 月週一凌晨 03:00，氣溫極低，大眾運輸停止，單車需求接近零",
    features: {
      yr: 0,
      mnth: 1,
      hr: 3,
      weekday: 1,
      holiday: 0,
      workingday: 1,
      season: 1,
      weathersit: 1,
      temp: 0.16,
      atemp: 0.18,
      hum: 0.65,
      windspeed: 0.10,
    },
  },
];
