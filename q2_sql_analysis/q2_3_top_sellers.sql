USE olist;

-- -------------------------------------------------------------
-- Step 1. Filter orders purchased in 2017.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_orders_2017;

CREATE TEMPORARY TABLE tmp_orders_2017 (
    order_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
    customer_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    order_status VARCHAR(20),
    order_delivered_customer_date DATETIME,
    order_estimated_delivery_date DATETIME,
    PRIMARY KEY (order_id),
    KEY idx_tmp_orders_customer (customer_id)
) ENGINE=InnoDB;

INSERT INTO tmp_orders_2017 (
    order_id,
    customer_id,
    order_status,
    order_delivered_customer_date,
    order_estimated_delivery_date
)
SELECT
    order_id,
    customer_id,
    order_status,
    order_delivered_customer_date,
    order_estimated_delivery_date
FROM orders
WHERE order_purchase_timestamp >= '2017-01-01'
  AND order_purchase_timestamp <  '2018-01-01';

-- -------------------------------------------------------------
-- Step 2. Build one row per seller-order combination.
--         Keep all order statuses here because cancellation rate
--         requires canceled orders as well as delivered orders.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_seller_order_base;

CREATE TEMPORARY TABLE tmp_seller_order_base (
    seller_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
    seller_state CHAR(2),
    order_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
    customer_unique_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    order_status VARCHAR(20),
    order_delivered_customer_date DATETIME,
    order_estimated_delivery_date DATETIME,
    total_items INT UNSIGNED NOT NULL,
    order_gmv DECIMAL(14,2) NOT NULL,
    PRIMARY KEY (seller_id, order_id),
    KEY idx_tmp_sob_order (order_id),
    KEY idx_tmp_sob_status (order_status)
) ENGINE=InnoDB;

INSERT INTO tmp_seller_order_base (
    seller_id,
    seller_state,
    order_id,
    customer_unique_id,
    order_status,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    total_items,
    order_gmv
)
SELECT
    oi.seller_id,
    s.seller_state,
    o.order_id,
    c.customer_unique_id,
    o.order_status,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    COUNT(*) AS total_items,
    SUM(COALESCE(oi.price, 0) + COALESCE(oi.freight_value, 0)) AS order_gmv
FROM tmp_orders_2017 AS o
INNER JOIN order_items AS oi
    ON o.order_id = oi.order_id
INNER JOIN sellers AS s
    ON oi.seller_id = s.seller_id
LEFT JOIN customers AS c
    ON o.customer_id = c.customer_id
GROUP BY
    oi.seller_id,
    s.seller_state,
    o.order_id,
    c.customer_unique_id,
    o.order_status,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date;

-- -------------------------------------------------------------
-- Step 3. Deduplicate reviews for 2017 orders using a CTE.
--         Keep the latest review for each order.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_reviews_2017;

CREATE TEMPORARY TABLE tmp_reviews_2017 (
    order_id VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
    review_score TINYINT,
    review_creation_date DATETIME,
    review_answer_timestamp DATETIME,
    PRIMARY KEY (order_id)
) ENGINE=InnoDB;

INSERT INTO tmp_reviews_2017 (
    order_id,
    review_score,
    review_creation_date,
    review_answer_timestamp
)
WITH ranked_reviews AS (
    SELECT
        r.order_id,
        r.review_score,
        r.review_creation_date,
        r.review_answer_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY r.order_id
            ORDER BY
                r.review_creation_date DESC,
                r.review_answer_timestamp DESC,
                r.review_id DESC
        ) AS rn
    FROM order_reviews AS r
    INNER JOIN tmp_orders_2017 AS o
        ON r.order_id = o.order_id
)
SELECT
    order_id,
    review_score,
    review_creation_date,
    review_answer_timestamp
FROM ranked_reviews
WHERE rn = 1;

-- -------------------------------------------------------------
-- Step 4. Calculate seller-level sales and operational metrics.
--         Sellers with no delivered orders are retained temporarily
--         for a complete status-based cancellation calculation;
--         they are excluded from the final sales ranking.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_seller_metrics;

CREATE TEMPORARY TABLE tmp_seller_metrics ENGINE=InnoDB AS
SELECT
    seller_id,
    seller_state,

    ROUND(SUM(
        CASE WHEN order_status = 'delivered'
             THEN order_gmv ELSE 0 END
    ), 2) AS total_gmv,

    SUM(
        CASE WHEN order_status = 'delivered'
             THEN 1 ELSE 0 END
    ) AS total_orders,

    SUM(
        CASE WHEN order_status = 'delivered'
             THEN total_items ELSE 0 END
    ) AS total_items,

    COUNT(DISTINCT
        CASE WHEN order_status = 'delivered'
             THEN customer_unique_id END
    ) AS unique_customers,

    ROUND(
        100.0 * SUM(
            CASE WHEN order_status = 'canceled'
                 THEN 1 ELSE 0 END
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS cancel_rate,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN order_status = 'delivered'
                 AND order_delivered_customer_date IS NOT NULL
                 AND order_estimated_delivery_date IS NOT NULL
                 AND order_delivered_customer_date > order_estimated_delivery_date
                THEN 1 ELSE 0
            END
        ) / NULLIF(
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                     AND order_delivered_customer_date IS NOT NULL
                     AND order_estimated_delivery_date IS NOT NULL
                    THEN 1 ELSE 0
                END
            ),
            0
        ),
        2
    ) AS late_delivery_rate
FROM tmp_seller_order_base
GROUP BY seller_id, seller_state;

ALTER TABLE tmp_seller_metrics
    ADD PRIMARY KEY (seller_id),
    ADD KEY idx_tmp_seller_metrics_gmv (total_gmv);

-- -------------------------------------------------------------
-- Step 5. Calculate seller-level review metrics.
--         Each deduplicated order review is associated with every
--         seller participating in that delivered order.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_seller_reviews;

CREATE TEMPORARY TABLE tmp_seller_reviews ENGINE=InnoDB AS
SELECT
    b.seller_id,
    ROUND(AVG(r.review_score), 2) AS avg_review_score,
    COUNT(r.review_score) AS reviewed_orders
FROM tmp_seller_order_base AS b
LEFT JOIN tmp_reviews_2017 AS r
    ON b.order_id = r.order_id
WHERE b.order_status = 'delivered'
GROUP BY b.seller_id;

ALTER TABLE tmp_seller_reviews
    ADD PRIMARY KEY (seller_id);

-- -------------------------------------------------------------
-- Step 6. Combine metrics and rank sellers.
--         Only sellers with at least one delivered seller-order
--         are eligible for the GMV ranking.
-- -------------------------------------------------------------
DROP TEMPORARY TABLE IF EXISTS tmp_final_sellers;

CREATE TEMPORARY TABLE tmp_final_sellers ENGINE=InnoDB AS
SELECT
    sm.seller_id,
    sm.seller_state,
    sm.total_orders,
    sm.total_items,
    sm.unique_customers,
    sm.total_gmv,
    sr.avg_review_score,
    COALESCE(sr.reviewed_orders, 0) AS reviewed_orders,
    sm.late_delivery_rate,
    sm.cancel_rate,

    ROW_NUMBER() OVER (
        ORDER BY sm.total_gmv DESC, sm.seller_id
    ) AS seller_rank,

    RANK() OVER (
        PARTITION BY sm.seller_state
        ORDER BY sm.total_gmv DESC
    ) AS state_rank

FROM tmp_seller_metrics AS sm
LEFT JOIN tmp_seller_reviews AS sr
    ON sm.seller_id = sr.seller_id
WHERE sm.total_orders > 0;

ALTER TABLE tmp_final_sellers
    ADD PRIMARY KEY (seller_id),
    ADD KEY idx_tmp_final_rank (seller_rank);

-- -------------------------------------------------------------
-- Step 7. Final output: Top 20 sellers by 2017 delivered GMV.
-- -------------------------------------------------------------
SELECT
    seller_rank,
    seller_id,
    seller_state,
    total_orders,
    total_items,
    unique_customers,
    total_gmv,
    avg_review_score,
    reviewed_orders,
    late_delivery_rate,
    cancel_rate,
    state_rank
FROM tmp_final_sellers
WHERE seller_rank <= 20
ORDER BY seller_rank;
