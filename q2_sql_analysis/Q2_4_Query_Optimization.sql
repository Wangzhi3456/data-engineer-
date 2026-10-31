
-- =========================================================
-- Q2.4: Query Optimization
-- Database: Olist
-- MySQL: 8.0+
-- Optimization: Temporary Tables + Indexes + Aggregation
-- =========================================================

USE olist;


-- =========================================================
-- STEP 1: Filter Delivered Orders in 2017
-- =========================================================

DROP TEMPORARY TABLE IF EXISTS tmp_delivered_2017;

CREATE TEMPORARY TABLE tmp_delivered_2017 AS
SELECT
    order_id,
    customer_id
FROM orders
WHERE order_status = 'delivered'
  AND order_purchase_timestamp >= '2017-01-01'
  AND order_purchase_timestamp < '2018-01-01';

-- Validation
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT order_id) AS unique_orders,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM tmp_delivered_2017;


-- =========================================================
-- STEP 2: Filter Target City Customers
-- =========================================================

DROP TEMPORARY TABLE IF EXISTS tmp_filtered_customers;

CREATE TEMPORARY TABLE tmp_filtered_customers AS
SELECT
    customer_id,
    customer_state
FROM customers
WHERE LOWER(customer_city) LIKE '%sao%';

-- Validation
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(DISTINCT customer_state) AS total_states
FROM tmp_filtered_customers;


-- =========================================================
-- STEP 3: Aggregate Order Revenue
-- Aggregate item prices by order before joining
-- =========================================================

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

-- Validation
SELECT
    COUNT(*) AS total_order_rows,
    COUNT(DISTINCT order_id) AS unique_orders,
    ROUND(SUM(revenue), 2) AS total_revenue
FROM tmp_order_revenue;


-- =========================================================
-- STEP 4: Deduplicate Reviews
-- Keep the latest review for each order
-- =========================================================

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

-- Validation
SELECT
    COUNT(*) AS total_reviewed_orders,
    COUNT(DISTINCT order_id) AS unique_orders,
    ROUND(AVG(review_score), 2) AS avg_review_score
FROM tmp_order_reviews;


-- =========================================================
-- STEP 5: Create Indexes on Temporary Tables
-- Use prefix indexes because ID columns are TEXT
-- =========================================================

ALTER TABLE tmp_delivered_2017
    ADD INDEX idx_tmp_d_customer_id (customer_id(32)),
    ADD INDEX idx_tmp_d_order_id (order_id(32));

ALTER TABLE tmp_filtered_customers
    ADD INDEX idx_tmp_c_customer_id (customer_id(32));

ALTER TABLE tmp_order_revenue
    ADD INDEX idx_tmp_oi_order_id (order_id(32));

ALTER TABLE tmp_order_reviews
    ADD INDEX idx_tmp_r_order_id (order_id(32));


-- =========================================================
-- STEP 6: Explain Optimized Query
-- Check whether indexes are used
-- =========================================================

EXPLAIN FORMAT=JSON
SELECT
    c.customer_state,
    COUNT(*) AS orders,
    ROUND(SUM(COALESCE(oi.revenue, 0)), 2) AS revenue,
    ROUND(AVG(r.review_score), 2) AS avg_score
FROM tmp_delivered_2017 AS d
INNER JOIN tmp_filtered_customers AS c
    ON d.customer_id = c.customer_id
LEFT JOIN tmp_order_revenue AS oi
    ON d.order_id = oi.order_id
LEFT JOIN tmp_order_reviews AS r
    ON d.order_id = r.order_id
GROUP BY c.customer_state;


-- =========================================================
-- STEP 7: Final Analysis by Customer State
-- =========================================================

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


-- =========================================================
-- STEP 8: Final Validation
-- =========================================================

SELECT
    SUM(orders) AS total_orders,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(*) AS total_states
FROM tmp_q24_final;
