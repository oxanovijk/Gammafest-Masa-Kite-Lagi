from low_leakage_pipeline_common import LowLeakageConfig, print_low_leakage_audit, run_low_leakage_pipeline


def main() -> None:
    config = LowLeakageConfig(
        name="ll3_outcome_preserving_scoreline_reranker",
        strategy_id="LL3",
        output_name="submission_ll3_outcome_preserving_reranker.csv",
        actions=["outcome_preserving_rerank"],
    )
    _, audit = run_low_leakage_pipeline(config)
    print_low_leakage_audit(audit)


if __name__ == "__main__":
    main()
