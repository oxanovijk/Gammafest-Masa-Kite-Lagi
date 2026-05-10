from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll7_small_train_derived_blend",
        strategy_id="LL7",
        output_name="submission_ll7_small_train_derived_blend.csv",
        pair_policy="prefer_draw",
        actions=["small_expert_blend"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
