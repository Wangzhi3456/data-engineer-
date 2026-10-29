import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 路径与日志配置
# ============================================================

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "archive" / "online_retail_II.csv"
OUTPUT_DIR = MODULE_DIR / "outputs"
LOG_PATH = MODULE_DIR / "pipeline_log.txt"

# 两张交易明细表使用相同的九个字段，退货表额外增加匹配发票号。
SALES_COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "unit_price",
    "total_amount",
    "invoice_datetime",
    "customer_id",
    "country",
]

# 经数据探索确认的非商品业务编码；保留原有过滤口径。
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES",
    "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "B", "C3",
    "TEST001", "TEST002",
}



def setup_logging() -> None:
    """配置控制台和文件日志。"""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# ============================================================
# Step 1：加载原始数据
# ============================================================

def load_data() -> pd.DataFrame:
    """
    读取 Online Retail II 原始数据。

    暂时保留原始字段及原始记录，不在读取阶段删除交易。
    """

    logger = logging.getLogger(__name__)

    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"找不到原始数据文件：{DATA_PATH}"
            )

        logger.info("开始读取原始数据：%s", DATA_PATH)

        data = pd.read_csv(
            DATA_PATH,
            dtype={
                "Invoice": "string",
                "StockCode": "string",
                "Description": "string",
                "Customer ID": "string",
                "Country": "string",
            },
            low_memory=False,
        )

        required_columns = {
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        }

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"原始数据缺少必要字段：{sorted(missing_columns)}"
            )

        logger.info(
            "原始数据读取完成：%s 行，%s 列",
            f"{len(data):,}",
            len(data.columns),
        )

        logger.info(
            "原始字段：%s",
            list(data.columns),
        )

        return data

    except Exception:
        logger.exception("加载原始数据失败")
        raise


# ============================================================
# Step 2：检查数据结构
# ============================================================

def inspect_data(data: pd.DataFrame) -> None:
    """检查原始数据规模、缺失情况和日期解析情况。"""

    logger = logging.getLogger(__name__)

    assert not data.empty, "原始数据为空"

    missing_summary = data.isna().sum()

    logger.info("各字段缺失数量：")

    for column, missing_count in missing_summary.items():
        logger.info(
            "%s：%s 条（%.4f%%）",
            column,
            f"{missing_count:,}",
            missing_count / len(data) * 100,
        )

    invoice_dates = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    invalid_dates = int(invoice_dates.isna().sum())

    logger.info(
        "无法解析的日期记录：%s 条",
        f"{invalid_dates:,}",
    )

    if invalid_dates:
        logger.warning(
            "存在无法解析的日期，后续需要明确处理策略"
        )

    logger.info(
        "日期范围：%s 至 %s",
        invoice_dates.min(),
        invoice_dates.max(),
    )

# ============================================================
# Step 3：规范化字段
# ============================================================

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """统一字段名称、数据类型及客户 ID 格式，保留原始交易记录。"""

    logger = logging.getLogger(__name__)

    try:
        work = data.copy()

        work = work.rename(
            columns={
                "Invoice": "invoice_no",
                "StockCode": "stock_code",
                "Description": "description",
                "Quantity": "quantity",
                "InvoiceDate": "invoice_datetime",
                "Price": "unit_price",
                "Customer ID": "customer_id",
                "Country": "country",
            }
        )

        work["invoice_no"] = work["invoice_no"].str.strip()
        work["stock_code"] = work["stock_code"].str.strip().str.upper()
        work["description"] = work["description"].str.strip()
        work["country"] = work["country"].str.strip()

        # Customer ID 是标识符，不是用于计算的数值。
        # 缺失值继续保留为 <NA>，不创建虚构客户。
        customer_id = work["customer_id"].str.strip()

        work["customer_id"] = customer_id.str.replace(
            r"\.0+$",
            "",
            regex=True,
        )

        work["invoice_datetime"] = pd.to_datetime(
            work["invoice_datetime"],
            errors="coerce",
        )

        work["quantity"] = pd.to_numeric(
            work["quantity"],
            errors="coerce",
        )

        work["unit_price"] = pd.to_numeric(
            work["unit_price"],
            errors="coerce",
        )

        work["total_amount"] = (
            work["quantity"] * work["unit_price"]
        )

        invalid_dates = int(work["invoice_datetime"].isna().sum())
        invalid_quantity = int(work["quantity"].isna().sum())
        invalid_price = int(work["unit_price"].isna().sum())

        logger.info(
            "字段规范化完成：日期无效 %s 条，数量无效 %s 条，单价无效 %s 条",
            f"{invalid_dates:,}",
            f"{invalid_quantity:,}",
            f"{invalid_price:,}",
        )

        # 此阶段只规范字段，不删除异常记录。
        assert len(work) == len(data), "字段规范化阶段不应改变记录数"
        assert work["total_amount"].notna().all(), (
            "total_amount 存在缺失值"
        )

        assert np.allclose(
            work["total_amount"],
            work["quantity"] * work["unit_price"],
        ), "total_amount 计算不正确"

        return work

    except Exception:
        logger.exception("字段规范化失败")
        raise

# ============================================================
# Step 4：筛选真实销售
# ============================================================

def filter_real_sales(data: pd.DataFrame) -> pd.DataFrame:
    """筛选正向商品销售，排除取消记录和已确认的非产品业务。"""

    logger = logging.getLogger(__name__)

    try:
        invoice_no = data["invoice_no"].str.strip().str.upper()
        stock_code = data["stock_code"].str.strip().str.upper()

        # 第一步：识别取消发票
        is_cancellation = invoice_no.str.startswith(
            "C", na=False
        )

        # 第二步：识别已确认的非产品业务代码
        is_non_product = stock_code.isin(NON_PRODUCT_CODES)

        # 第三步：保留正向销售数量
        has_positive_quantity = data["quantity"].gt(0)

        # 第四步：排除负单价
        has_nonnegative_price = data["unit_price"].ge(0)

        # 第五步：排除关键字段缺失或日期无效的记录
        has_required_values = (
            data["invoice_no"].notna()
            & data["invoice_no"].str.strip().ne("")
            & data["stock_code"].notna()
            & data["stock_code"].str.strip().ne("")
            & data["invoice_datetime"].notna()
            & data["quantity"].notna()
            & data["unit_price"].notna()
            & data["total_amount"].notna()
        )

        mask = (
            ~is_cancellation
            & ~is_non_product
            & has_positive_quantity
            & has_nonnegative_price
            & has_required_values
        )

        sales = data.loc[mask].copy()

        # 只输出任务指定的九个字段
        output_columns = SALES_COLUMNS

        sales = sales[output_columns].reset_index(drop=True)

        # 关键业务断言
        assert not sales.empty, "销售事实表为空"
        assert sales["invoice_no"].notna().all(), "发票号存在缺失"
        assert sales["stock_code"].notna().all(), "商品代码存在缺失"
        assert sales["quantity"].gt(0).all(), "存在非正销售数量"
        assert sales["unit_price"].ge(0).all(), "存在负单价"
        assert sales["total_amount"].ge(0).all(), "存在负交易金额"
        assert not sales["invoice_no"].str.upper().str.startswith(
            "C", na=False
        ).any(), "取消发票进入了销售事实表"
        assert not sales["stock_code"].str.upper().isin(
            NON_PRODUCT_CODES
        ).any(), "非产品业务代码进入了销售事实表"

        # 记录筛选结果，不生成额外的统计文件
        logger.info("销售筛选完成")
        logger.info("筛选前记录数：%s", f"{len(data):,}")
        logger.info("销售事实表记录数：%s", f"{len(sales):,}")
        logger.info(
            "排除记录数：%s",
            f"{len(data) - len(sales):,}",
        )
        logger.info(
            "销售事实表金额合计：%.2f",
            sales["total_amount"].sum(),
        )

        return sales

    except Exception:
        logger.exception("真实销售筛选失败")
        raise

# ============================================================
# Step 5：计算客户 RFM 特征
# ============================================================

def compute_rfm(
    sales: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    基于真实销售记录计算客户 RFM 特征。

    reference_date：整个原始数据集的最后一个自然日。
    缺失客户 ID 的记录不参与客户分群。
    """

    logger = logging.getLogger(__name__)

    try:
        if sales.empty:
            raise ValueError("销售事实表为空，无法计算 RFM")

        reference_date = pd.Timestamp(reference_date).normalize()

        # 1. 只保留可以识别客户的销售记录
        customer_sales = sales.loc[
            sales["customer_id"].notna()
            & sales["customer_id"].str.strip().ne("")
        ].copy()

        if customer_sales.empty:
            raise ValueError("没有有效客户 ID，无法计算 RFM")

        # 2. 确保日期和金额有效
        assert customer_sales["invoice_datetime"].notna().all()
        assert customer_sales["total_amount"].notna().all()
        assert customer_sales["total_amount"].ge(0).all()

        # 3. 按客户计算首次购买、最后购买、购买频次和消费金额
        rfm = (
            customer_sales.groupby("customer_id", as_index=False)
            .agg(
                first_purchase_date=("invoice_datetime", "min"),
                last_purchase_date=("invoice_datetime", "max"),
                frequency=("invoice_no", "nunique"),
                monetary=("total_amount", "sum"),
                unique_products=("stock_code", "nunique"),
            )
        )

        # 4. 以整个原始数据集的最后一天为基准计算 Recency
        rfm["recency_days"] = (
            reference_date
            - rfm["last_purchase_date"].dt.normalize()
        ).dt.days

        # 5. 统计每个客户的主要国家
        # 若多个国家出现次数相同，按国家名称升序确定结果，
        # 保证相同输入得到相同输出。
        country_counts = (
            customer_sales.groupby(
                ["customer_id", "country"],
                dropna=False,
            )
            .size()
            .rename("country_records")
            .reset_index()
        )

        country_counts = country_counts.sort_values(
            ["customer_id", "country_records", "country"],
            ascending=[True, False, True],
            na_position="last",
        )

        primary_country = (
            country_counts
            .drop_duplicates("customer_id", keep="first")
            [["customer_id", "country"]]
        )

        rfm = rfm.merge(
            primary_country,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        # 6. 标记一次性购买客户
        rfm["is_one_time_buyer"] = rfm["frequency"].eq(1)

        # 7. 按任务要求整理字段顺序
        output_columns = [
            "customer_id",
            "recency_days",
            "frequency",
            "monetary",
            "first_purchase_date",
            "last_purchase_date",
            "unique_products",
            "country",
            "is_one_time_buyer",
        ]

        rfm = rfm[output_columns].sort_values(
            "customer_id"
        ).reset_index(drop=True)

        # 8. 输出前校验
        assert not rfm.empty, "客户特征表为空"
        assert rfm["customer_id"].notna().all()
        assert rfm["customer_id"].is_unique, "客户 ID 不唯一"
        assert rfm["recency_days"].notna().all()
        assert rfm["recency_days"].ge(0).all(), (
            "recency_days 不能为负"
        )
        assert rfm["frequency"].gt(0).all()
        assert rfm["monetary"].ge(0).all()
        assert rfm["unique_products"].gt(0).all()
        assert (
            rfm["is_one_time_buyer"]
            == rfm["frequency"].eq(1)
        ).all()

        logger.info("客户 RFM 计算完成")
        logger.info("有效客户数：%s", f"{len(rfm):,}")
        logger.info(
            "一次性购买客户数：%s",
            f"{rfm['is_one_time_buyer'].sum():,}",
        )
        logger.info(
            "Recency 天数范围：%s 至 %s",
            rfm["recency_days"].min(),
            rfm["recency_days"].max(),
        )
        logger.info(
            "客户消费金额合计：%.2f",
            rfm["monetary"].sum(),
        )

        return rfm

    except Exception:
        logger.exception("客户 RFM 计算失败")
        raise

# ============================================================
# Step 6：提取退货记录并匹配原始销售
# ============================================================

def match_returns(
    data: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """提取 C 开头发票，并匹配最近且剩余数量足够的原始销售发票。"""

    logger = logging.getLogger(__name__)

    try:
        output_columns = SALES_COLUMNS

        # 1. 提取所有 C 开头的发票记录
        invoice_no = data["invoice_no"].str.strip().str.upper()

        returns = data.loc[
            invoice_no.str.startswith("C", na=False),
            output_columns,
        ].copy().reset_index(drop=True)

        if returns.empty:
            logger.warning("没有发现 C 开头的发票")
            returns["matched_original_invoice"] = pd.Series(
                dtype="string"
            )
            return returns

        # 2. 按客户、商品、原始发票和销售时间聚合可退数量
        # 只允许匹配有效客户的正向销售。
        eligible_sales = sales.loc[
            sales["customer_id"].notna()
            & sales["customer_id"].str.strip().ne("")
            & sales["quantity"].gt(0)
        ]

        original_lines = (
            eligible_sales.groupby(
                [
                    "customer_id",
                    "stock_code",
                    "invoice_no",
                    "invoice_datetime",
                ],
                as_index=False,
                sort=False,
            )
            .agg(quantity=("quantity", "sum"))
            .sort_values(
                ["customer_id", "stock_code", "invoice_datetime", "invoice_no"]
            )
        )

        # 每个客户和商品对应一组按时间升序排列的原始销售。
        sales_by_key = {}

        for row in original_lines.itertuples(index=False):
            key = (row.customer_id, row.stock_code)

            sales_by_key.setdefault(key, []).append({
                "invoice_no": row.invoice_no,
                "invoice_datetime": row.invoice_datetime,
                "remaining_quantity": int(row.quantity),
            })

        returns["matched_original_invoice"] = pd.Series(
            pd.NA,
            index=returns.index,
            dtype="string",
        )

        matched_count = 0

        # 3. 按退货时间顺序匹配，避免后续退货重复使用已退数量。
        ordered_returns = returns.sort_values(
            ["invoice_datetime", "invoice_no", "stock_code"]
        )

        for row in ordered_returns.itertuples():
            # 只匹配负数量退货记录
            if row.quantity >= 0:
                continue

            # 缺少客户 ID 时无法可靠匹配
            if pd.isna(row.customer_id) or not row.customer_id.strip():
                continue

            return_quantity = abs(int(row.quantity))
            key = (row.customer_id, row.stock_code)

            candidates = sales_by_key.get(key, [])

            # 候选销售按时间升序排列，反向检查最近的原始销售。
            for candidate in reversed(candidates):

                # 原始销售必须严格早于退货
                if candidate["invoice_datetime"] >= row.invoice_datetime:
                    continue

                # 单笔原始发票必须有足够的剩余可退数量
                if candidate["remaining_quantity"] < return_quantity:
                    continue

                returns.at[
                    row.Index,
                    "matched_original_invoice",
                ] = candidate["invoice_no"]

                candidate["remaining_quantity"] -= return_quantity
                matched_count += 1

                # 每条退货只匹配一张原始发票
                break

        # 4. 校验输出结构
        returns = returns[
            output_columns + ["matched_original_invoice"]
        ]

        assert returns["invoice_no"].str.upper().str.startswith(
            "C", na=False
        ).all(), "退货日志包含非 C 开头发票"

        assert len(returns) == int(
            invoice_no.str.startswith("C", na=False).sum()
        ), "退货日志记录数不正确"

        negative_returns = returns["quantity"].lt(0)
        unmatched_negative = (
            negative_returns
            & returns["matched_original_invoice"].isna()
        )

        logger.info("退货匹配完成")
        logger.info("退货日志记录数：%s", f"{len(returns):,}")
        logger.info(
            "负数量退货记录数：%s",
            f"{int(negative_returns.sum()):,}",
        )
        logger.info(
            "成功匹配记录数：%s",
            f"{matched_count:,}",
        )
        logger.info(
            "未匹配负数量退货数：%s",
            f"{int(unmatched_negative.sum()):,}",
        )

        return returns.reset_index(drop=True)

    except Exception:
        logger.exception("退货匹配失败")
        raise

# ============================================================
# Step 7：验证退货匹配结果
# ============================================================

def validate_return_matches(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict:
    """检查退货匹配的时间、身份、商品及数量约束。"""

    logger = logging.getLogger(__name__)

    try:
        matched = returns.loc[
            returns["matched_original_invoice"].notna()
        ].copy()

        # 1. 所有匹配成功的记录必须是真正的负数量退货
        assert matched["quantity"].lt(0).all(), (
            "发现非负数量记录被匹配"
        )

        # 2. 汇总原始发票中每个客户、商品的可销售数量
        original = (
            sales.loc[sales["customer_id"].notna()]
            .groupby(
                ["customer_id", "stock_code", "invoice_no"],
                as_index=False,
            )
            .agg(
                original_quantity=("quantity", "sum"),
                original_datetime=("invoice_datetime", "min"),
            )
            .rename(
                columns={"invoice_no": "matched_original_invoice"}
            )
        )

        # 3. 将已匹配退货与原始销售关联
        checked = matched.merge(
            original,
            on=[
                "customer_id",
                "stock_code",
                "matched_original_invoice",
            ],
            how="left",
            validate="many_to_one",
        )

        assert checked["original_quantity"].notna().all(), (
            "存在找不到原始销售的匹配结果"
        )

        # 4. 检查原始销售时间必须早于退货时间
        invalid_time = (
            checked["original_datetime"]
            >= checked["invoice_datetime"]
        ).sum()

        assert invalid_time == 0, (
            f"发现 {invalid_time} 条时间顺序错误的匹配"
        )

        # 5. 检查累计退货数量不能超过原始销售数量
        returned_quantities = (
            checked.assign(
                returned_quantity=checked["quantity"].abs()
            )
            .groupby(
                [
                    "customer_id",
                    "stock_code",
                    "matched_original_invoice",
                ],
                as_index=False,
            )
            .agg(
                returned_quantity=("returned_quantity", "sum"),
                original_quantity=("original_quantity", "first"),
            )
        )

        over_returned = (
            returned_quantities["returned_quantity"]
            > returned_quantities["original_quantity"]
        ).sum()

        assert over_returned == 0, (
            f"发现 {over_returned} 组累计退货数量超限"
        )

        # 6. 统计未匹配记录
        negative_returns = returns["quantity"].lt(0)

        unmatched = returns.loc[
            negative_returns
            & returns["matched_original_invoice"].isna()
        ]

        missing_customer = (
            unmatched["customer_id"].isna()
            | unmatched["customer_id"].str.strip().eq("").fillna(False)
        ).sum()

        # 7. 统计有效客户、商品组合是否存在销售候选
        original_keys = (
            sales.loc[sales["customer_id"].notna(),
                      ["customer_id", "stock_code"]]
            .drop_duplicates()
            .assign(has_candidate=True)
        )

        unmatched_checked = unmatched.merge(
            original_keys,
            on=["customer_id", "stock_code"],
            how="left",
            validate="many_to_one",
        )

        no_candidate = (
            unmatched_checked["customer_id"].notna()
            & unmatched_checked["has_candidate"].isna()
        ).sum()

        other_unmatched = (
            len(unmatched)
            - int(missing_customer)
            - int(no_candidate)
        )

        result = {
            "matched_count": int(len(matched)),
            "unmatched_count": int(len(unmatched)),
            "invalid_time_count": int(invalid_time),
            "over_returned_count": int(over_returned),
            "missing_customer_count": int(missing_customer),
            "no_candidate_count": int(no_candidate),
            "other_unmatched_count": int(other_unmatched),
        }

        logger.info("退货匹配质量验证通过")
        logger.info("成功匹配：%s", f"{result['matched_count']:,}")
        logger.info("未匹配：%s", f"{result['unmatched_count']:,}")
        logger.info("时间顺序错误：%s", result["invalid_time_count"])
        logger.info("累计退货超限：%s", result["over_returned_count"])
        logger.info(
            "未匹配：客户 ID 缺失 %s 条",
            result["missing_customer_count"],
        )
        logger.info(
            "未匹配：没有客户-商品销售候选 %s 条",
            result["no_candidate_count"],
        )
        logger.info(
            "未匹配：其他原因 %s 条",
            result["other_unmatched_count"],
        )

        return result

    except Exception:
        logger.exception("退货匹配质量验证失败")
        raise


# ============================================================
# Step 8：运行 ETL 流程
# ============================================================

def main() -> None:
    start_time = time.perf_counter()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # ====================================================
        # 1. 读取、检查并规范化原始数据
        # ====================================================
        data = load_data()
        inspect_data(data)
        data = prepare_data(data)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # ====================================================
        # 2. 生成销售事实表
        # ====================================================
        sales_facts = filter_real_sales(data)

        sales_facts.to_parquet(
            OUTPUT_DIR / "sales_facts.parquet",
            index=False,
        )

        logger.info(
            "销售事实表已保存：%s",
            OUTPUT_DIR / "sales_facts.parquet",
        )

        # ====================================================
        # 3. 生成客户 RFM 特征表
        # ====================================================
        reference_date = data["invoice_datetime"].max().normalize()

        logger.info(
            "RFM 基准日期：%s",
            reference_date.date(),
        )

        customer_features = compute_rfm(
            sales_facts,
            reference_date,
        )

        customer_features.to_parquet(
            OUTPUT_DIR / "customer_features.parquet",
            index=False,
        )

        logger.info(
            "客户特征表已保存：%s",
            OUTPUT_DIR / "customer_features.parquet",
        )

        # ====================================================
        # 4. 生成退货日志
        # ====================================================
        returns_log = match_returns(
            data=data,
            sales=sales_facts,
        )

        # ====================================================
        # 5. 验证退货匹配结果
        # ====================================================
        validate_return_matches(
            sales=sales_facts,
            returns=returns_log,
        )

        # 验证通过后才保存退货日志
        returns_log.to_parquet(
            OUTPUT_DIR / "returns_log.parquet",
            index=False,
        )

        logger.info(
            "退货日志已保存：%s",
            OUTPUT_DIR / "returns_log.parquet",
        )

        # ====================================================
        # 6. 最终运行时间与性能检查
        # ====================================================
        elapsed = time.perf_counter() - start_time

        logger.info(
            "ETL Pipeline 全部完成，耗时 %.2f 秒",
            elapsed,
        )

        if elapsed >= 120:
            logger.warning(
                "运行时间超过题目要求的 120 秒限制"
            )
        else:
            logger.info(
                "性能要求验证通过：运行时间低于 120 秒"
            )

    except Exception:
        logger.exception("ETL 流程执行失败")
        raise


if __name__ == "__main__":
    main()
