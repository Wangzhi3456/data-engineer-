USE olist;

-- Q2.1: Identify product categories without English translations
-- Distinguish missing category values from missing translations.

SELECT
    p.product_category_name,

    COUNT(*) AS product_count,

    CASE
        WHEN p.product_category_name IS NULL
            THEN 'Missing Category'
        ELSE 'Missing English Translation'
    END AS issue_type

FROM products p

LEFT JOIN product_category_name_translation t
    ON p.product_category_name = t.product_category_name

WHERE t.product_category_name IS NULL

GROUP BY
    p.product_category_name

ORDER BY product_count DESC;