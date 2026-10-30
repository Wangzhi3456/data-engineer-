# Q2.4 - 慢查询优化分析

## 原始 SQL

```sql
USE olist;

EXPLAIN FORMAT=JSON

SELECT 
    c.customer_state,
    COUNT(DISTINCT o.order_id) AS orders,
    SUM(oi.price) AS revenue,
    AVG(r.review_score) AS avg_score
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp >= '2017-01-01'
  AND o.order_purchase_timestamp < '2018-01-01'
  AND LOWER(c.customer_city) LIKE '%sao%'
GROUP BY c.customer_state
ORDER BY revenue DESC;
```

查询目标：

- 筛选 2017 年已交付订单。
- 筛选城市名称包含 `sao` 的客户。
- 按客户所在州统计订单数量、商品收入和平均评价分数。
- 按商品收入从高到低排序。

---

## 第 1 问：性能问题分析（至少 3 个）

### 问题 1：多表 JOIN 缺少有效索引，导致查询效率下降

原始查询需要关联 orders、customers、order_items 和 order_reviews 四张表，主要通过 customer_id 和 order_id 进行关联。

通过 `SHOW INDEX` 和 `information_schema.STATISTICS` 检查发现：

- orders 表已有 order_id 和 customer_id 索引。
- customers 表已有 customer_id 索引。
- order_items 表已有 order_id 索引。
- order_reviews 表原先缺少有效的 order_id 索引。

此外，相关 ID 字段在数据导入时被设置为 TEXT 类型，直接创建普通索引会触发 Error Code 1170，需要使用指定长度的前缀索引。

**优化方向：** 为缺少索引的原始表关联字段建立索引，并为后续临时表中的关联字段建立合适的索引，减少 JOIN 时的数据扫描成本。

### 问题 2：LOWER() 函数和前置通配符影响索引使用

原始查询使用：

```sql
LOWER(c.customer_city) LIKE '%sao%'
```

这里存在两个问题：

1. `LOWER()` 对字段进行函数转换，普通索引难以直接利用该表达式进行定位。
2. `LIKE '%sao%'` 使用前置通配符，无法通过普通 B-tree 索引进行有效的前缀范围查找。

因此，客户城市筛选可能需要扫描客户表中的大量记录。

**优化方向：** 将客户筛选独立为一个步骤，提前生成符合条件的客户临时表，避免在最终多表 JOIN 过程中重复执行筛选条件。

### 问题 3：多表 JOIN 导致中间结果膨胀

原始查询直接关联 order_items 和 order_reviews。

由于一个订单可能包含多个商品，也可能存在多条评价，两张明细表直接关联会产生乘积式数据膨胀。

例如，一个订单包含 3 件商品和 2 条评价，关联后可能产生 6 行记录。

这不仅增加了 JOIN 的计算成本，还会造成商品金额重复累计。

通过 EXPLAIN FORMAT=JSON 检查，原始查询存在较高的中间结果估算量，并涉及 Hash Join、临时表和排序操作。

**优化方向：** 在最终 JOIN 之前，分别对商品金额和评价数据进行预聚合及去重，使每个订单在关联时最多对应一条商品汇总记录和一条评价记录。

---

## 第 2 问：优化方案

### 优化后的 SQL

结合此前 Q2.3 的实际开发经验，最初计划使用 CTE 整合订单筛选、客户筛选、商品汇总和评价处理，但复杂 CTE 整合后曾出现查询超时和连接断开问题。

因此，本题最终采用 TEMPORARY TABLE 分步骤执行，将复杂查询拆分为独立的数据处理任务，并在临时表创建后建立关联索引。

这种方式能够分阶段执行和验证中间结果，也方便定位具体步骤的性能问题。

完整优化 SQL 如下：

```sql
USE olist;

-- =====================================================
-- Step 1: Filter Delivered Orders in 2017
-- =====================================================

DROP TEMPORARY TABLE IF EXISTS tmp_delivered_2017;

CREATE TEMPORARY TABLE tmp_delivered_2017 AS
SELECT
    order_id,
    customer_id
FROM orders
WHERE order_status = 'delivered'
  AND order_purchase_timestamp >= '2017-01-01'
  AND order_purchase_timestamp < '2018-01-01';


-- =====================================================
-- Step 2: Filter Target City Customers
-- =====================================================

DROP TEMPORARY TABLE IF EXISTS tmp_filtered_customers;

CREATE TEMPORARY TABLE tmp_filtered_customers AS
SELECT
    customer_id,
    customer_state
FROM customers
WHERE LOWER(customer_city) LIKE '%sao%';


-- =====================================================
-- Step 3: Aggregate Order Revenue
-- =====================================================

DROP TEMPORARY TABLE IF EXISTS tmp_order_revenue;

CREATE TEMPORARY TABLE tmp_order_revenue AS
SELECT
    oi.order_id,
    SUM(COALESCE(oi.price, 0)) AS revenue
FROM order_items AS oi
INNER JOIN tmp_delivered_2017 AS d
    ON oi.order_id = d.order_id
INNER JOIN tmp_filtered_customers AS c
    ON d.customer_id = c.customer_id
GROUP BY oi.order_id;


-- =====================================================
-- Step 4: Deduplicate Reviews
-- Keep the Latest Review for Each Order
-- =====================================================

DROP TEMPORARY TABLE IF EXISTS tmp_order_reviews;

CREATE TEMPORARY TABLE tmp_order_reviews AS
SELECT
    order_id,
    review_score
FROM (
    SELECT
        r.order_id,
        r.review_score,

        ROW_NUMBER() OVER (
            PARTITION BY r.order_id
            ORDER BY
                r.review_creation_date DESC,
                r.review_answer_timestamp DESC,
                r.review_id DESC
        ) AS rn

    FROM order_reviews AS r

    INNER JOIN tmp_order_revenue AS rev
        ON r.order_id = rev.order_id
) AS ranked_reviews
WHERE rn = 1;


-- =====================================================
-- Step 5: Create Indexes on Temporary Tables
-- TEXT fields require prefix indexes
-- =====================================================

ALTER TABLE tmp_delivered_2017
    ADD INDEX idx_tmp_d_customer_id (customer_id(32)),
    ADD INDEX idx_tmp_d_order_id (order_id(32));

ALTER TABLE tmp_filtered_customers
    ADD INDEX idx_tmp_c_customer_id (customer_id(32));

ALTER TABLE tmp_order_revenue
    ADD INDEX idx_tmp_oi_order_id (order_id(32));

ALTER TABLE tmp_order_reviews
    ADD INDEX idx_tmp_r_order_id (order_id(32));


-- =====================================================
-- Step 6: Final Analysis by Customer State
-- =====================================================

DROP TEMPORARY TABLE IF EXISTS tmp_q24_final;

CREATE TEMPORARY TABLE tmp_q24_final AS
SELECT
    c.customer_state,

    COUNT(*) AS orders,

    ROUND(
        SUM(COALESCE(oi.revenue, 0)),
        2
    ) AS revenue,

    ROUND(
        AVG(r.review_score),
        2
    ) AS avg_score

FROM tmp_delivered_2017 AS d

INNER JOIN tmp_filtered_customers AS c
    ON d.customer_id = c.customer_id

LEFT JOIN tmp_order_revenue AS oi
    ON d.order_id = oi.order_id

LEFT JOIN tmp_order_reviews AS r
    ON d.order_id = r.order_id

GROUP BY c.customer_state;


-- Final Output

SELECT
    customer_state,
    orders,
    revenue,
    avg_score
FROM tmp_q24_final
ORDER BY revenue DESC;


-- =====================================================
-- Step 7: Final Validation
-- =====================================================

SELECT
    SUM(orders) AS total_orders,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(*) AS total_states
FROM tmp_q24_final;
```
### 原始表索引设计与验证

针对原始表的关联字段和筛选字段，进行了以下索引优化：

```sql
USE olist;

-- Reviews table
CREATE INDEX idx_reviews_order_id
ON order_reviews(order_id(32));

-- Orders table
CREATE INDEX idx_orders_status_purchase
ON orders(order_status(20), order_purchase_timestamp);

ANALYZE TABLE orders;

### 优化前 vs 优化后实测对比

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| EXPLAIN Query Cost | 66,450.89 | 24,665.30 |
| Orders 表访问方式 | ALL | 临时表通过 ref 关联 |
| 商品表关联方式 | ref | ref |
| Reviews 表关联方式 | ref | ref |
| 中间结果处理 | 多表直接 JOIN | 分步筛选、预聚合及去重 |
| 最终查询执行时间 | 未单独记录 | 0.954 秒 |
| 五步临时表创建时间 | — | 3.969 秒 |
| 四张临时表建立索引时间 | — | 3.000 秒 |
| 已记录操作耗时合计 | — | 7.923 秒 |

注：

1. 优化前 Query Cost 为原始查询在建立相关索引后的执行计划记录；优化后 Query Cost 为临时表建立关联索引后的最终查询执行计划记录。两者执行结构不同，成本仅作为优化器估算参考。
2. 优化前原始完整查询的实际执行时间没有取得，因此不计算实际加速倍数。
3. 优化后的 7.923 秒为已记录的五步临时表创建、索引建立和最终查询耗时之和，不代表整个开发过程的总耗时。
4. 优化后的执行时间为 MySQL Workbench 实际记录的最终 SELECT 查询耗时。

---

## 第 3 问：潜在的统计错误（关键题）

> `SUM(oi.price)` 的结果可能不是你以为的「订单总收入」。为什么？

原始查询直接关联订单表、商品表和评价表，但这三张表的统计粒度不同：

- orders：一笔订单一条记录。
- order_items：一笔订单可能对应多个商品记录。
- order_reviews：一笔订单可能对应多条评价记录。

当一个订单同时存在多条商品记录和多条评价记录时，直接 JOIN 会产生数据膨胀。

例如：

| 订单 | 商品数量 | 评价数量 | JOIN 后行数 |
|---|---:|---:|---:|
| A | 3 | 2 | 6 |
| B | 2 | 1 | 2 |

订单 A 原本只有 3 条商品记录，关联评价后变成 6 条，导致每件商品的价格都被重复累计。

因此，原始 SQL 中的：

```sql
SUM(oi.price)
```

可能高估实际商品销售收入。

同时：

```sql
AVG(r.review_score)
```

也可能受到评价重复记录的影响，无法准确代表每笔订单的平均评价水平。

### 实际数据验证

通过统计订单、商品和评价的关联结果，进一步验证数据膨胀问题。

第一步：检查原始评价表。

| 检查项目 | 实际结果 |
|---|---:|
| Reviews 原始记录 | 99,224 |
| 唯一评价订单 | 98,673 |
| 多余评价记录 | 551 |

第二步：检查 2017 年目标订单的关联情况。

| 检查项目 | 实际结果 |
|---|---:|
| 2017 年目标订单 | 43,428 |
| 商品记录关联评价前 | 49,556 |
| 商品记录关联评价后 | 49,948 |
| 评价造成的额外 JOIN 行 | 392 |
| 存在多条评价的订单 | 315 |

第三步：检查 JOIN 后的数据粒度。

| 检查项目 | 实际结果 |
|---|---:|
| JOIN 后记录数 | 49,948 |
| 唯一订单数 | 43,428 |
| 有商品记录的订单 | 43,428 |
| 有评价记录的订单 | 43,100 |
| 每笔订单平均关联行数 | 1.15 |

这些结果证明，原始查询确实存在实际的数据膨胀风险，而不仅仅是理论上的问题。

**修正方案：**

采用先聚合、后关联的方式，分别处理商品收入和评价数据。

1. 商品表：先按 order_id 汇总商品金额，每个订单只保留一条收入记录。
2. 评价表：使用 ROW_NUMBER() 对同一订单的评价按照时间排序，只保留最新一条。
3. 最终查询：将已经处理到订单粒度的数据进行关联，避免商品金额重复累计。

商品收入修正：

```sql
CREATE TEMPORARY TABLE tmp_order_revenue AS
SELECT
    oi.order_id,
    SUM(COALESCE(oi.price, 0)) AS revenue
FROM order_items oi
INNER JOIN tmp_delivered_2017 d
    ON oi.order_id = d.order_id
INNER JOIN tmp_filtered_customers c
    ON d.customer_id = c.customer_id
GROUP BY oi.order_id;
```

评价去重修正：

```sql
CREATE TEMPORARY TABLE tmp_order_reviews AS
SELECT
    order_id,
    review_score
FROM (
    SELECT
        r.order_id,
        r.review_score,
        ROW_NUMBER() OVER (
            PARTITION BY r.order_id
            ORDER BY
                r.review_creation_date DESC,
                r.review_answer_timestamp DESC,
                r.review_id DESC
        ) AS rn
    FROM order_reviews r
    INNER JOIN tmp_order_revenue rev
        ON r.order_id = rev.order_id
) ranked_reviews
WHERE rn = 1;
```

最终关联：

```sql
SELECT
    c.customer_state,
    COUNT(*) AS orders,
    ROUND(SUM(COALESCE(oi.revenue, 0)), 2) AS revenue,
    ROUND(AVG(r.review_score), 2) AS avg_score
FROM tmp_delivered_2017 d
INNER JOIN tmp_filtered_customers c
    ON d.customer_id = c.customer_id
LEFT JOIN tmp_order_revenue oi
    ON d.order_id = oi.order_id
LEFT JOIN tmp_order_reviews r
    ON d.order_id = r.order_id
GROUP BY c.customer_state
ORDER BY revenue DESC;
```

### 最终结果验证

| 指标 | 实际结果 |
|---|---:|
| 2017 年已交付订单 | 43,428 |
| 目标城市客户 | 21,092 |
| 目标订单数量 | 8,524 |
| 有评价的目标订单 | 8,458 |
| 最终统计州数量 | 22 |
| 总商品收入 | 1,062,522.57 |
| 平均评价分数 | 4.21 |

最终 22 个州的订单数量合计为 8,524，商品收入合计为 1,062,522.57，与商品收入预聚合结果完全一致。

这说明最终关联没有造成商品收入重复累计，评价缺失的 66 笔订单也仍然保留在订单和收入统计中。

---

## 总结

本题通过执行计划分析、索引检查、数据粒度排查和分阶段 SQL 改写，解决了原始多表 JOIN 查询中的主要性能问题与统计准确性风险。

本次实践的主要成果：

1. **查询性能优化：** 通过原始表索引、临时表关联索引和分阶段处理，降低最终查询的估算成本。
2. **统计准确性优化：** 识别出商品明细与评价明细直接 JOIN 造成的数据膨胀问题，通过商品收入预聚合和评价去重解决重复累计。
3. **结果验证：** 最终获得 8,524 笔订单、1,062,522.57 的商品收入及 22 个州的统计结果，且收入与预聚合结果一致。

本次实践表明，数据库优化不能只关注 SQL 是否能够成功执行，还需要同时考虑查询执行计划、索引设计、关联数据粒度以及最终统计结果的正确性。