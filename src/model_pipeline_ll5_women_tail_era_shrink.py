from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll5_women_tail_era_shrink",
        strategy_id="LL5",
        output_name="submission_ll5_women_tail_era_shrink.csv",
        actions=["women_tail_era_shrink"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
