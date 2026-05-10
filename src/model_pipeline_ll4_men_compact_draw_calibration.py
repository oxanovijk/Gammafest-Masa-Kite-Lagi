from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll4_men_compact_draw_calibration",
        strategy_id="LL4",
        output_name="submission_ll4_men_compact_draw_calibration.csv",
        actions=["draw_calibration"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
