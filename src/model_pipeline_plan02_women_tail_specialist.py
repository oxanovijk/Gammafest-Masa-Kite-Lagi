from pattern_pipeline_common import SegmentExpertConfig, print_audit, run_segment_expert_strategy


def main() -> None:
    config = SegmentExpertConfig(
        name="plan02_women_tail_specialist",
        output_name="submission_plan02_women_tail_specialist.csv",
        selector_levels=[
            (["gender", "tournament", "era"], 20),
            (["gender", "tournament"], 60),
            (["gender", "archetype"], 80),
            (["gender"], 1),
        ],
        transform_groups=["tail"],
        transform_min_n=20,
    )
    _, audit = run_segment_expert_strategy(config)
    print_audit(audit)


if __name__ == "__main__":
    main()
