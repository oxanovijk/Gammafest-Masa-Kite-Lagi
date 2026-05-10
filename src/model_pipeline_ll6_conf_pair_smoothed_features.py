from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll6_conf_pair_smoothed_feature_calibration",
        strategy_id="LL6",
        output_name="submission_ll6_conf_pair_smoothed_features.csv",
        actions=["conf_pair_smoothed"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
