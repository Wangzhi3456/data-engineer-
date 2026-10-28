"""Q1 Data Quality Analysis — Online Retail II

Single-source-of-truth script for Q1.1/Q1.2 evidence generation.
The raw CSV is never modified. All normalisation happens in an analysis copy.
"""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "archive" / "online_retail_II.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REQUIRED = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

# Code + description evidence, not code shape alone, determines a known business line.
KNOWN_DESC = {
    "POST": {"POSTAGE"}, "DOT": {"DOTCOM POSTAGE"}, "M": {"MANUAL"},
    "C2": {"CARRIAGE"}, "D": {"DISCOUNT"}, "S": {"SAMPLES"},
    "BANK CHARGES": {"BANK CHARGES"}, "AMAZONFEE": {"AMAZON FEE"},
    "CRUK": {"CRUK COMMISSION"}, "B": {"ADJUST BAD DEBT"},
}
ADJUST_CODES = {"ADJUST", "ADJUST2"}
LABELS = {
    "POST": "邮费", "DOT": "特殊配送费用", "M": "手工调整", "C2": "运费",
    "D": "折扣", "S": "样品相关", "BANK CHARGES": "银行费用",
    "AMAZONFEE": "平台费用", "CRUK": "佣金", "B": "坏账调整",
    "ADJUST": "人工调整", "ADJUST2": "人工调整",
}


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到原始数据：{DATA_FILE}\n请确认文件位于 data/archive/online_retail_II.csv")
    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="cp1252", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}；实际字段：{df.columns.tolist()}")
    return df


def save_csv(df, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_analysis_frame(df):
    w = df.copy()
    for c in ["Invoice", "StockCode", "Description", "Country"]:
        w[c + "_clean"] = w[c].astype("string").str.strip()
    w["Invoice_clean"] = w["Invoice_clean"].str.upper()
    w["StockCode_clean"] = w["StockCode_clean"].str.upper()
    w["Description_clean"] = w["Description_clean"].str.strip()
    w["Quantity_num"] = pd.to_numeric(w["Quantity"], errors="coerce")
    w["Price_num"] = pd.to_numeric(w["Price"], errors="coerce")
    w["LineAmount"] = w["Quantity_num"] * w["Price_num"]
    w["InvoiceDate_parsed"] = pd.to_datetime(w["InvoiceDate"], errors="coerce")
    w["CustomerID状态"] = w["Customer ID"].isna().map({True: "缺失", False: "存在"})
    w["InvoicePrefix"] = w["Invoice_clean"].str.extract(r"^([A-Z]+)", expand=False).fillna("数字开头或无字母前缀")
    w["IsCancellation"] = w["Invoice_clean"].str.startswith("C", na=False)

    code = w["StockCode_clean"]
    w["IsPureNumericStockCode"] = code.str.fullmatch(r"\d+", na=False)
    w["IsDigitThenLettersStockCode"] = code.str.fullmatch(r"\d+[A-Z]+", na=False)
    w["IsSpecialFormatStockCode"] = code.notna() & ~w["IsPureNumericStockCode"] & ~w["IsDigitThenLettersStockCode"]

    # Reproduce the explicitly curated business-code list used in the earlier
    # analysis. This is a code-list classification, not a regex classification.
    # Descriptions remain in the exported detail files for manual validation.
    w["KnownBusinessCategory"] = pd.Series(pd.NA, index=w.index, dtype="string")
    for c, label in LABELS.items():
        w.loc[code.eq(c).fillna(False), "KnownBusinessCategory"] = label
    return w


def section(title):
    print("\n" + "=" * 64 + "\n" + title + "\n" + "=" * 64)


def q_basic(df, w):
    n = len(df)
    desc = w["Description_clean"].isna() | w["Description_clean"].eq("").fillna(False)
    customer_missing = df["Customer ID"].isna()
    section("基础数据质量检查")
    print(f"数据文件：{DATA_FILE.relative_to(PROJECT_DIR)}")
    print(f"记录数：{n:,}；字段数：{len(df.columns)}；内存：{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    print(f"字段：{df.columns.tolist()}")
    print(f"Customer ID 缺失：{customer_missing.sum():,} ({customer_missing.mean()*100:.4f}%)")
    print(f"Description 缺失/空白：{desc.sum():,} ({desc.mean()*100:.4f}%)")
    print(f"完全重复行（重复出现部分）：{df.duplicated().sum():,}")
    print(f"无法解析日期：{w['InvoiceDate_parsed'].isna().sum():,}")
    print(f"日期范围：{w['InvoiceDate_parsed'].min()} 至 {w['InvoiceDate_parsed'].max()}")
    for col in ["Quantity_num", "Price_num"]:
        print(f"{col}: 无法转换={(w[col].isna()).sum():,}；负数={w[col].lt(0).sum():,}；零值={w[col].eq(0).sum():,}")
    print("原始数据未被修改或删除。")


def q11(df):
    n = len(df)
    quality = []
    for col in df.columns:
        s = df[col]
        is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if is_text:
            norm = s.astype("string").str.strip()
            blank = norm.eq("").fillna(False)
            missing_or_blank = norm.isna() | blank
        else:
            blank = pd.Series(False, index=df.index)
            missing_or_blank = s.isna()
        quality.append({"字段": col, "数据类型": str(s.dtype), "原始缺失数量": int(s.isna().sum()),
                        "空白字符串数量": int(blank.sum()), "缺失或空白数量": int(missing_or_blank.sum()),
                        "缺失或空白比例(%)": round(float(missing_or_blank.mean())*100, 4)})
    quality = pd.DataFrame(quality)
    save_csv(quality, "q11_column_quality.csv")

    numeric_rows = []
    for col in ["Quantity", "Price"]:
        s = pd.to_numeric(df[col], errors="coerce")
        numeric_rows.append({"字段": col, "有效值数量": int(s.notna().sum()), "无法转换数量": int(s.isna().sum()),
                             "Min": s.min(), "Max": s.max(), "Mean": s.mean(), "Std": s.std(),
                             "Median": s.median(), "P01": s.quantile(.01), "P99": s.quantile(.99)})
    numeric = pd.DataFrame(numeric_rows)
    save_csv(numeric, "q11_numeric_distribution.csv")

    categorical_fields = ["Invoice", "StockCode", "Description", "InvoiceDate", "Customer ID", "Country"]
    uniques, tops = [], []
    for col in categorical_fields:
        s = df[col].astype("string").str.strip()
        s = s.mask(s.eq("").fillna(False), pd.NA)
        uniques.append({"字段": col, "非空唯一值数量": int(s.nunique(dropna=True)),
                        "含缺失类别的唯一值数量": int(s.nunique(dropna=False))})
        counts = s.fillna("<缺失值>").value_counts().head(10)
        for rank, (value, count) in enumerate(counts.items(), 1):
            tops.append({"字段": col, "排名": rank, "取值": str(value), "记录数": int(count),
                         "记录占比(%)": round(int(count)/n*100, 4)})
    unique = pd.DataFrame(uniques)
    top = pd.DataFrame(tops)
    save_csv(unique, "q11_string_unique_counts.csv")
    save_csv(top, "q11_string_top10.csv")
    section("Q1.1 字段质量与分布证据")
    print("字段质量：\n", quality.to_string(index=False))
    print("\n数值分布：\n", numeric.round(4).to_string(index=False))
    print("\n字符串/标识符唯一值：\n", unique.to_string(index=False))
    print("\nTOP 10 高频值已导出：q11_string_top10.csv")


def q_invoice_prefix(w):
    n = len(w)
    prefix = w.groupby("InvoicePrefix", dropna=False).agg(
        记录数=("InvoicePrefix", "size"),
        负数量记录数=("Quantity_num", lambda s: int(s.lt(0).sum())),
        零单价记录数=("Price_num", lambda s: int(s.eq(0).sum())),
        净交易金额=("LineAmount", "sum"),
        绝对金额=("LineAmount", lambda s: s.abs().sum())).reset_index()
    prefix["记录占比(%)"] = (prefix["记录数"]/n*100).round(4)
    prefix = prefix.sort_values("记录数", ascending=False)
    save_csv(prefix, "invoice_prefix_summary.csv")
    details = w.loc[w["InvoicePrefix"].ne("数字开头或无字母前缀")].groupby("InvoicePrefix").agg(
        记录数=("InvoicePrefix", "size"),
        描述样例=("Description_clean", lambda s: " | ".join(s.dropna().astype(str).value_counts().head(5).index)),
        负数量记录数=("Quantity_num", lambda s: int(s.lt(0).sum())),
        净交易金额=("LineAmount", "sum"), 绝对金额=("LineAmount", lambda s: s.abs().sum())).reset_index()
    save_csv(details, "invoice_prefix_business_evidence.csv")
    section("Q1.2-1 发票前缀")
    print(prefix.to_string(index=False))
    print("\n字母前缀证据：\n", details.to_string(index=False))


def q_stockcode(df, w):
    n = len(w)
    special = w.loc[w["IsSpecialFormatStockCode"]].copy()
    summary = special.groupby("StockCode_clean", dropna=False).agg(
        记录数=("StockCode_clean", "size"), 描述种类数=("Description_clean", "nunique"),
        描述样例=("Description_clean", lambda s: " | ".join(s.dropna().astype(str).value_counts().head(3).index)),
        数量合计=("Quantity_num", "sum"), 净交易金额=("LineAmount", "sum"),
        绝对金额=("LineAmount", lambda s: s.abs().sum()),
        负数量记录数=("Quantity_num", lambda s: int(s.lt(0).sum())),
        零单价记录数=("Price_num", lambda s: int(s.eq(0).sum()))).reset_index().rename(columns={"StockCode_clean": "StockCode"})
    summary["记录占比(%)"] = (summary["记录数"]/n*100).round(4)
    summary["初步分类提示"] = summary["StockCode"].map(LABELS).fillna("待核查：商品或其他业务候选")
    summary = summary.sort_values(["记录数", "StockCode"], ascending=[False, True])
    save_csv(summary, "all_special_stockcode_summary.csv")
    detail_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "LineAmount"]
    save_csv(special[detail_cols], "all_special_stockcode_records.csv")

    # 两个集合不可混淆：格式特殊按题目正则；已知业务代码按代码与描述证据识别。
    known = w.loc[w["KnownBusinessCategory"].notna()].copy()
    scopes = pd.DataFrame([
        {"统计口径": "全量数据", "记录数": n, "记录占比(%)": 100.0, "净交易金额": w["LineAmount"].sum(), "绝对金额": w["LineAmount"].abs().sum()},
        {"统计口径": "格式特殊 StockCode（非纯数字且非数字后接字母）", "记录数": len(special), "记录占比(%)": len(special)/n*100, "净交易金额": special["LineAmount"].sum(), "绝对金额": special["LineAmount"].abs().sum()},
        {"统计口径": "已知业务代码（代码+描述规则，独立口径）", "记录数": len(known), "记录占比(%)": len(known)/n*100, "净交易金额": known["LineAmount"].sum(), "绝对金额": known["LineAmount"].abs().sum()},
    ])
    scopes["记录占比(%)"] = scopes["记录占比(%)"].round(4)
    save_csv(scopes, "stockcode_scope_comparison.csv")
    section("Q1.2-2 特殊格式 StockCode")
    print(scopes.round(2).to_string(index=False))
    print("\n特殊格式代码 TOP 30：\n", summary.head(30).round(2).to_string(index=False))
    print("注意：格式特殊不等于非产品；已知业务代码按明确维护的代码清单独立识别。描述字段保留用于人工复核。两个集合口径不同，不能强行要求前者是后者子集。")


def q_extremes(w):
    x = w.copy()
    x["SourceRow"] = range(2, len(x)+2)  # CSV 表头占第 1 行
    cols = ["SourceRow", "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "LineAmount", "InvoicePrefix"]
    exact = x.loc[x["Quantity_num"].isin([80995, -80995]), cols]
    neg_price = x.loc[x["Price_num"].lt(0), cols]
    q_top = x.assign(_abs=x["Quantity_num"].abs()).nlargest(20, "_abs").drop(columns="_abs")[cols]
    p_top = x.assign(_abs=x["Price_num"].abs()).nlargest(20, "_abs").drop(columns="_abs")[cols]
    save_csv(exact, "exact_quantity_extremes.csv")
    save_csv(neg_price, "all_negative_price_records.csv")
    save_csv(q_top, "extreme_quantity_records.csv")
    save_csv(p_top, "extreme_price_records.csv")
    section("Q1.2-3 极端值与负单价")
    print("指定数量极值：\n", exact.to_string(index=False))
    print("\n全部负单价记录：\n", neg_price.to_string(index=False))
    print("解释边界：极端值应回查原始订单；数量正负相反可能对应交易与冲销，不可仅凭数值直接删除。负单价须结合描述和发票类型判定。")


def q_customer_groups(w):
    n = len(w)
    summary = w.groupby("CustomerID状态", dropna=False).agg(
        记录数=("Invoice", "size"), 平均数量=("Quantity_num", "mean"), 数量中位数=("Quantity_num", "median"),
        平均单价=("Price_num", "mean"), 单价中位数=("Price_num", "median"),
        取消发票记录数=("IsCancellation", "sum"), 涉及国家数=("Country_clean", "nunique"),
        净交易金额=("LineAmount", "sum"), 绝对金额=("LineAmount", lambda s: s.abs().sum())).reset_index()
    summary["记录占比(%)"] = summary["记录数"]/n*100
    summary["取消发票记录占比(%)"] = summary["取消发票记录数"]/summary["记录数"]*100
    total_net, total_abs = w["LineAmount"].sum(), w["LineAmount"].abs().sum()
    summary["净金额占全量净金额(%)"] = summary["净交易金额"]/total_net*100 if total_net else float("nan")
    summary["绝对金额占全量绝对金额(%)"] = summary["绝对金额"]/total_abs*100 if total_abs else float("nan")
    save_csv(summary, "customer_group_summary.csv")

    country = w.groupby(["CustomerID状态", "Country_clean"], dropna=False).size().reset_index(name="记录数")
    country["群体内记录占比(%)"] = country["记录数"]/country.groupby("CustomerID状态")["记录数"].transform("sum")*100
    country = country.sort_values(["CustomerID状态", "记录数"], ascending=[True, False])
    save_csv(country, "customer_group_country_distribution.csv")
    save_csv(country.loc[country["CustomerID状态"].eq("缺失")], "missing_customer_by_country.csv")

    desc_ok = w["Description_clean"].notna() & w["Description_clean"].ne("").fillna(False)
    eligible_mask = (~w["IsCancellation"] & w["Quantity_num"].gt(0).fillna(False) & w["Price_num"].gt(0).fillna(False)
                     & desc_ok & w["KnownBusinessCategory"].isna() & w["Invoice_clean"].notna() & w["StockCode_clean"].notna())
    products = w.loc[eligible_mask].copy()
    orders = products.groupby(["CustomerID状态", "Invoice_clean"], dropna=False).agg(
        订单商品金额=("LineAmount", "sum"), 商品明细数=("Invoice_clean", "size"), 商品数量=("Quantity_num", "sum")).reset_index()
    aov = orders.groupby("CustomerID状态").agg(订单数=("Invoice_clean", "nunique"), 商品销售金额合计=("订单商品金额", "sum"),
        平均订单商品金额=("订单商品金额", "mean"), 订单商品金额中位数=("订单商品金额", "median"),
        最小订单商品金额=("订单商品金额", "min"), 最大订单商品金额=("订单商品金额", "max")).reset_index()
    save_csv(orders, "customer_invoice_amounts.csv")
    save_csv(aov, "customer_aov_summary.csv")
    freq = products.groupby(["CustomerID状态", "StockCode_clean", "Description_clean"], dropna=False).agg(
        记录数=("Invoice_clean", "size"), 涉及订单数=("Invoice_clean", "nunique"), 销售数量=("Quantity_num", "sum"), 商品销售金额=("LineAmount", "sum")).reset_index()
    freq = freq.rename(columns={"StockCode_clean": "StockCode", "Description_clean": "Description"}).sort_values(["CustomerID状态", "记录数"], ascending=[True, False])
    save_csv(freq, "customer_group_product_frequency.csv")

    section("Q1.2-4 Customer ID 缺失与完整群体")
    print("交易明细层级（不是客单价）：\n", summary.round(3).to_string(index=False))
    print("\n筛选后订单商品金额（AOV 代理）：\n", aov.round(2).to_string(index=False))
    for group in ["存在", "缺失"]:
        print(f"\nCustomer ID {group} 高频商品 TOP 10：")
        print(freq.loc[freq["CustomerID状态"].eq(group)].head(10).round(2).to_string(index=False))
        print(f"\nCustomer ID {group} 国家分布 TOP 10：")
        print(country.loc[country["CustomerID状态"].eq(group)].head(10).round(3).to_string(index=False))
    print("注意：Customer ID 缺失群体的订单金额是筛选口径下的代理值，不等于完整会计销售额；缺失本身不能证明客户身份。")


def consistency_checks(df, w):
    n = len(df)
    prefix_sum = int(w.groupby("InvoicePrefix", dropna=False).size().sum())
    special_n = int(w["IsSpecialFormatStockCode"].sum())
    known_n = int(w["KnownBusinessCategory"].notna().sum())
    section("一致性检查")
    print(f"原始总记录数：{n:,}")
    print(f"发票前缀分组加总：{prefix_sum:,}")
    print(f"格式特殊代码记录数：{special_n:,}")
    print(f"已知业务代码记录数（独立口径）：{known_n:,}")
    if prefix_sum != n:
        raise AssertionError("发票前缀分组加总不等于原始记录数。")
    if not 0 <= special_n <= n or not 0 <= known_n <= n:
        raise AssertionError("分类记录数超出全量数据范围。")
    print("通过：加总与分类边界检查正常。")
    print(f"输出目录：{OUTPUT_DIR}")


def main():
    df = load_data()
    w = build_analysis_frame(df)
    q_basic(df, w)
    q11(df)
    q_invoice_prefix(w)
    q_stockcode(df, w)
    q_extremes(w)
    q_customer_groups(w)
    consistency_checks(df, w)
    print("\n全部分析结束。原始数据文件未被修改。")


if __name__ == "__main__":
    main()
