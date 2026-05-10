from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll1_train_archetype_prior",
        strategy_id="LL1",
        output_name="submission_ll1_train_archetype_prior.csv",
        actions=["archetype_prior"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
