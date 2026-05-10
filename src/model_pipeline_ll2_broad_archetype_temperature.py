from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll2_broad_archetype_temperature_proxy",
        strategy_id="LL2",
        output_name="submission_ll2_broad_archetype_temperature.csv",
        actions=["temperature_proxy"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
