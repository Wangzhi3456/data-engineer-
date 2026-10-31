"""
项目统一运行入口（run_all.py）

运行方式：
    python run_all.py

功能说明：
    本文件用于统一组织和运行项目中支持 Python 自动执行的模块，
    包括 Q1 数据质量分析、Q3 ETL Pipeline，以及满足运行条件时的 Q4 LLM 抽取任务。

运行限制：
    1. Q2 SQL 与业务分析需要提前启动 MySQL 8.0+ 数据库，
       并完成 Olist 数据导入，相关 SQL 文件需在数据库环境中单独执行。
    2. Q4 LLM 抽取需要提前配置 API Key、模型接口及相关依赖，
       否则无法完成实际模型调用。
    3. Q5 为系统设计文档，不涉及 Python 程序执行。
    4. Q6 为加分分析文档，不属于自动执行的代码模块。
    5. 本项目数据集需要用户提前下载并放置在指定的 data/ 目录，
       本入口不负责自动下载数据集。

注意：
    本入口并非所有题目文件的无条件一键执行工具。
    各模块仍可根据实际环境和需求独立运行。
"""


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODULES = {
    "Q1": ROOT / "q1_data_quality" / "q1_code.py",
    "Q3": ROOT / "q3_etl_pipeline" / "pipeline.py",
    "Q4": ROOT / "q4_llm_extraction" / "extract_reviews.py",
}

REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "REFLECTION.md",
    ROOT / "TOOLS.md",
    ROOT / "AI_LOG.md",
    ROOT / "requirements.txt",

    ROOT / "q1_data_quality" / "data_quality_report.md",
    ROOT / "q1_data_quality" / "findings.md",
    ROOT / "q1_data_quality" / "cleaning_strategy.md",
    ROOT / "q1_data_quality" / "q1_code.py",

    ROOT / "q2_sql_analysis" / "q2_1_translation_gap.sql",
    ROOT / "q2_sql_analysis" / "q2_2_monthly_transaction_analysis.sql",
    ROOT / "q2_sql_analysis" / "q2_3_top_sellers.sql",
    ROOT / "q2_sql_analysis" / "Q2_4_Query_Optimization.sql",

    ROOT / "q3_etl_pipeline" / "pipeline.py",
    ROOT / "q3_etl_pipeline" / "validation_report.md",

    ROOT / "q4_llm_extraction" / "extract_reviews.py",
    ROOT / "q4_llm_extraction" / "prompt_template.txt",
    ROOT / "q4_llm_extraction" / "extracted_issues.json",
    ROOT / "q4_llm_extraction" / "pipeline_design.md",
    ROOT / "q4_llm_extraction" / "cost_report.md",
    ROOT / "q4_llm_extraction" / "accuracy_evaluation.md",

    ROOT / "q5_system_design" / "real_time_pipeline_design.md",
]

Q3_OUTPUTS = [
    ROOT / "q3_etl_pipeline" / "outputs" / "sales_facts.parquet",
    ROOT / "q3_etl_pipeline" / "outputs" / "customer_features.parquet",
    ROOT / "q3_etl_pipeline" / "outputs" / "returns_log.parquet",
]

DATA_FILES = [
    ROOT / "data" / "archive" / "online_retail_II.csv",
    ROOT / "data" / "olist" / "olist_orders_dataset.csv",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_project() -> bool:
    print_header("PROJECT STRUCTURE CHECK")

    missing = []

    for path in REQUIRED_FILES:
        if path.exists():
            print(f"[OK] {path.relative_to(ROOT)}")
        else:
            print(f"[MISSING] {path.relative_to(ROOT)}")
            missing.append(path)

    print_header("DATASET CHECK")

    missing_data = []

    for path in DATA_FILES:
        if path.exists() and path.stat().st_size > 0:
            print(f"[OK] {path.relative_to(ROOT)}")
        else:
            print(f"[MISSING OR EMPTY] {path.relative_to(ROOT)}")
            missing_data.append(path)

    print_header("Q3 OUTPUT CHECK")

    for path in Q3_OUTPUTS:
        status = "EXISTS" if path.exists() else "NOT GENERATED"
        print(f"[{status}] {path.relative_to(ROOT)}")

    if missing or missing_data:
        print(f"\nMissing required files: {len(missing)}")
        print(f"Missing datasets: {len(missing_data)}")
        return False

    print("\nProject structure check passed.")
    return True


def run_module(name: str, script: Path) -> bool:
    print_header(f"RUNNING {name}")

    if not script.exists():
        print(f"[FAILED] Script not found: {script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
        )

        if result.returncode == 0:
            print(f"\n[{name}] COMPLETED")
            return True

        print(f"\n[{name}] FAILED: exit code {result.returncode}")
        return False

    except Exception as exc:
        print(f"\n[{name}] ERROR: {exc}")
        return False


def validate_q3_outputs() -> bool:
    print_header("Q3 OUTPUT VALIDATION")

    missing = []

    for path in Q3_OUTPUTS:
        if path.exists() and path.stat().st_size > 0:
            print(f"[OK] {path.name}")
        else:
            print(f"[MISSING OR EMPTY] {path.name}")
            missing.append(path)

    return not missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified entry point for the Data Engineer assessment."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check project structure and files.",
    )

    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Explicitly run Q4 DeepSeek extraction. May incur API costs.",
    )

    args = parser.parse_args()

    print_header("DATA ENGINEER PROJECT PIPELINE")

    print(f"Project root: {ROOT}")
    print(f"Python: {sys.executable}")

    if args.check:
        return 0 if check_project() else 1

    if not check_project():
        print("\n[ERROR] Project structure check failed.")
        return 1

    results = {}

    # Q1: Data quality analysis
    results["Q1"] = run_module("Q1", MODULES["Q1"])

    # Q2: SQL scripts are delivered for database execution.
    print_header("Q2 SQL ANALYSIS")
    print("SQL scripts are available for manual execution.")
    print("Automatic database execution is not enabled.")
    results["Q2"] = None

    # Q3: ETL pipeline
    results["Q3"] = run_module("Q3", MODULES["Q3"])

    if results["Q3"]:
        results["Q3_OUTPUTS"] = validate_q3_outputs()
    else:
        results["Q3_OUTPUTS"] = False

    # Q4: LLM extraction is opt-in.
    print_header("Q4 LLM EXTRACTION")

    if args.include_llm:
        results["Q4"] = run_module("Q4", MODULES["Q4"])
    else:
        print("SKIPPED: DeepSeek API is disabled by default.")
        print("Use --include-llm to explicitly enable it.")
        results["Q4"] = None

    # Q5: System design document
    print_header("Q5 SYSTEM DESIGN")

    q5_doc = ROOT / "q5_system_design" / "real_time_pipeline_design.md"
    results["Q5"] = q5_doc.exists()

    print("[OK] Q5 document found" if results["Q5"]
          else "[FAILED] Q5 document missing")

    # Final summary
    print_header("FINAL EXECUTION SUMMARY")

    failed = False

    for name, status in results.items():
        if status is True:
            label = "PASSED"
        elif status is None:
            label = "SKIPPED"
        else:
            label = "FAILED"
            failed = True

        print(f"{name:<15}: {label}")

    print("\nPipeline finished.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())