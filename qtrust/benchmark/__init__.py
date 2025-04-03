"""
QTrust Benchmark Package

Package chứa các công cụ benchmark và so sánh hiệu suất của QTrust
với các hệ thống blockchain và phi blockchain khác.
"""

__version__ = '0.1.0'

from qtrust.benchmark.benchmark_scenarios import (
    BenchmarkScenario,
    NetworkCondition,
    AttackProfile,
    WorkloadProfile,
    NodeProfile,
    get_scenario,
    get_all_scenario_ids,
    get_all_scenarios
)

from qtrust.benchmark.benchmark_runner import (
    run_benchmark,
    run_all_benchmarks,
    generate_comparison_report,
    plot_comparison_charts
) 