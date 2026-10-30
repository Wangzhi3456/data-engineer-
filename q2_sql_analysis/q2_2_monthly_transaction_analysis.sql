
-- Q2.2: Monthly Platform Actual Transaction Analysis

USE olist;

-- Aggregate order item amounts first to avoid duplicate orders
WITH order_item_totals AS (
    SELECT
        order_id,
        SUM(
            COALESCE(price, 0) + COALESCE(freight_value, 0)
        ) AS order_gmv
    FROM order_items
    GROUP BY order_id
)
SELECT
    -- 1. Order month
    DATE_FORMAT(
        o.order_purchase_timestamp,
        '%Y-%m'
    ) AS month,

    -- 2. Total orders
    COUNT(*) AS total_orders,

    -- 3. Delivered orders
    SUM(
        CASE
            WHEN o.order_status = 'delivered'
            THEN 1 ELSE 0
        END
    ) AS delivered_orders,

    -- 4. Canceled orders
    SUM(
        CASE
            WHEN o.order_status = 'canceled'
            THEN 1 ELSE 0
        END
    ) AS canceled_orders,

    -- 5. Cancellation rate (%)
    ROUND(
        SUM(
            CASE
                WHEN o.order_status = 'canceled'
                THEN 1 ELSE 0
            END
        ) / COUNT(*) * 100,
        2
    ) AS cancel_rate,

    -- 6. Delivered GMV (product price + freight)
    ROUND(
        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(i.order_gmv, 0)
                ELSE 0
            END
        ),
        2
    ) AS delivered_gmv,

    -- 7. Average delivery days
    ROUND(
        AVG(
            CASE
                WHEN o.order_status = 'delivered'
                 AND o.order_delivered_customer_date IS NOT NULL
                THEN DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_purchase_timestamp
                )
            END
        ),
        2
    ) AS avg_delivery_days

FROM orders AS o

LEFT JOIN order_item_totals AS i
    ON o.order_id = i.order_id

GROUP BY DATE_FORMAT(
    o.order_purchase_timestamp,
    '%Y-%m'
)

ORDER BY month;
