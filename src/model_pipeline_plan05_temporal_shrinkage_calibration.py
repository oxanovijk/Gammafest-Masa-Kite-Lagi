from pattern_pipeline_common import SegmentExpertConfig, print_audit, run_segment_expert_strategy


def main() -> None:
    config = SegmentExpertConfig(
        name="plan05_temporal_shrinkage_calibration",
        output_name="submission_plan05_temporal_shrinkage_calibration.csv",
        selector_levels=[
            (["gender", "tournament", "era"], 20),
            (["gender", "tournament"], 100),
            (["gender", "archetype"], 100),
            (["gender"], 1),
        ],
        transform_groups=["tail", "compact", "temporal"],
        transform_min_n=20,
    )
    _, audit = run_segment_expert_strategy(config)
    print_audit(audit)


if __name__ == "__main__":
    main()
