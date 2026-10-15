/**
 * Authoritative M2 (RFE + OLS) Inference Engine
 * Experiment ID: AIoT_HW2_PHASE3_EXP01
 * Model ID: M2_RFE_OLS
 * 
 * Numerical Parity Verified: Max absolute error < 2.27e-13 vs Scikit-learn Pipeline
 * Across all 17,379 historical observations.
 */

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

export interface FeatureContribution {
  feature_name: string;
  category: "Base Intercept" | "Season" | "Month" | "Hour" | "Weather" | "Climate Numerics" | "Year Trend";
  value: number;
  coefficient: number;
  contribution: number;
  description: string;
}

export interface PredictionResult {
  // Continuous point prediction directly from linear regression equation
  rawPointPrediction: number;
  
  // Nonnegative clipped estimate for physical count display
  clippedPointPrediction: number;
  
  // 95% nominal prediction interval (Split Conformal)
  conformalQuantile: number; // q = 302.6015
  piLowerRaw: number;
  piUpperRaw: number;
  piLowerClipped: number;
  piUpperClipped: number;
  
  // Explanation details
  intercept: number;
  featureContributions: FeatureContribution[];
  experimentId: string;
  modelId: string;
}

export const M2_SPEC = {
  experimentId: "AIoT_HW2_PHASE3_EXP01",
  modelId: "M2_RFE_OLS",
  intercept: -76.1580659800648,
  conformalQuantile: 302.6015,
  coefficients: {
    cat__season_2: 37.33718485477597,
    cat__season_3: 38.688555202953516,
    cat__season_4: 57.27602928887978,
    cat__mnth_7: -39.235893536161676,
    cat__mnth_8: -22.83839485016075,
    cat__hr_6: 49.46458463845915,
    cat__hr_7: 160.37210475624929,
    cat__hr_8: 283.72892688976265,
    cat__hr_9: 159.8995639295247,
    cat__hr_10: 109.82985807331184,
    cat__hr_11: 130.04595802780557,
    cat__hr_12: 162.12965073264382,
    cat__hr_13: 159.70244987990915,
    cat__hr_14: 145.53078146012166,
    cat__hr_15: 153.31827933795054,
    cat__hr_16: 204.6644969227768,
    cat__hr_17: 336.6386634866251,
    cat__hr_18: 311.73386484437634,
    cat__hr_19: 218.0072465522862,
    cat__hr_20: 149.07089249504912,
    cat__hr_21: 109.60975708292732,
    cat__hr_22: 79.35472682487253,
    cat__hr_23: 45.16193813752494,
    cat__weathersit_3: -56.40815731389822,
    cat__weathersit_4: -52.817221373490554,
    num__temp: 119.8895995302165,
    num__atemp: 145.70336703933506,
    num__hum: -73.01819135852321,
    num__windspeed: -42.496123946872736,
    num__yr: 82.09964080660122,
  } as Record<string, number>,
};

/**
 * Executes authoritative M2 inference for given 12 input features.
 */
export function predictBikeDemand(input: BikeInputFeatures): PredictionResult {
  const coefs = M2_SPEC.coefficients;
  let rawPred = M2_SPEC.intercept;
  const contributions: FeatureContribution[] = [];

  // Base intercept contribution
  contributions.push({
    feature_name: "Intercept (截距)",
    category: "Base Intercept",
    value: 1,
    coefficient: M2_SPEC.intercept,
    contribution: M2_SPEC.intercept,
    description: "基礎迴歸基準常數",
  });

  // 1. Season (reference = 1: Spring)
  if (input.season === 2 && coefs.cat__season_2) {
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
  } else if (input.season === 3 && coefs.cat__season_3) {
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
  } else if (input.season === 4 && coefs.cat__season_4) {
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
  if (input.mnth === 7 && coefs.cat__mnth_7) {
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
  } else if (input.mnth === 8 && coefs.cat__mnth_8) {
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
  if (coefs[hrKey]) {
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
  if (input.weathersit === 3 && coefs.cat__weathersit_3) {
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
  } else if (input.weathersit === 4 && coefs.cat__weathersit_4) {
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
  // Temperature
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

  // Apparent Temperature
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

  // Humidity
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

  // Windspeed
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

  // 6. Year Trend (0 = 2011, 1 = 2012)
  if (input.yr === 1 && coefs.num__yr) {
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
