from pattern_pipeline_common import SegmentExpertConfig, print_audit, run_segment_expert_strategy


def main() -> None:
    config = SegmentExpertConfig(
        name="plan01_balanced_segment_prior_reranker",
        output_name="submission_plan01_balanced_segment_prior_reranker.csv",
        selector_levels=[
            (["gender", "tournament", "era"], 80),
            (["gender", "tournament"], 100),
            (["gender", "archetype"], 100),
            (["gender"], 1),
        ],
        transform_groups=["tail", "compact"],
        transform_min_n=80,
    )
    _, audit = run_segment_expert_strategy(config)
    print_audit(audit)


if __name__ == "__main__":
    main()
