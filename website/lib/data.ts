import finalTestMetrics from "../data/final_test_metrics.json";
import predictionIntervalMetrics from "../data/prediction_interval_metrics.json";
import bestModelMetadata from "../data/best_model_metadata.json";
import phase3ExperimentMetadata from "../data/phase3_experiment_metadata.json";
import datasetMetadata from "../data/dataset_metadata.json";
import modelComparison from "../data/model_comparison.json";
import selectedFeatures from "../data/selected_features.json";
import groupedFeatureAblation from "../data/grouped_feature_ablation.json";
import hourlyCoverageDiagnostic from "../data/hourly_coverage_diagnostic.json";
import testTimeline7d from "../data/test_timeline_7d.json";
import testScatterSamples from "../data/test_scatter_samples.json";
import residualDistributionBins from "../data/residual_distribution_bins.json";
import m2ModelSpec from "../data/m2_model_spec.json";

export {
  finalTestMetrics,
  predictionIntervalMetrics,
  bestModelMetadata,
  phase3ExperimentMetadata,
  datasetMetadata,
  modelComparison,
  selectedFeatures,
  groupedFeatureAblation,
  hourlyCoverageDiagnostic,
  testTimeline7d,
  testScatterSamples,
  residualDistributionBins,
  m2ModelSpec,
};

export interface ModelComparisonRow {
  model_id: string;
  model_name: string;
  best_hyperparameters: string;
  selected_features_count: number;
  training_mae: number;
  training_mse: number;
  training_rmse: number;
  training_r2: number;
  cv_mae_mean: number;
  cv_mse_mean: number;
  cv_rmse_mean: number;
  cv_rmse_std: number;
  cv_r2_mean: number;
}

export interface SelectedFeatureRow {
  model_id: string;
  encoded_feature_name: string;
  original_feature_name: string;
  selection_status: string;
  coefficient: number | null;
}

export interface GroupedAblationRow {
  dropped_group: string;
  cv_rmse: number;
  delta_rmse: number;
}

export interface HourlyCoverageRow {
  hr: number;
  empirical_coverage: number;
  coverage_pct: number;
  sample_count: number;
  actual_mean: number;
  pred_mean: number;
}

export interface TimelinePoint {
  timestamp: string;
  actual_cnt: number;
  predicted_cnt: number;
  pi_lower_raw: number;
  pi_upper_raw: number;
  pi_lower_clipped: number;
  pi_upper_clipped: number;
  residual: number;
  covered: boolean;
}

export interface ScatterSample {
  actual: number;
  predicted: number;
  residual: number;
  timestamp: string;
  covered: boolean;
}

export interface ResidualBin {
  bin_start: number;
  bin_end: number;
  bin_center: number;
  count: number;
}
