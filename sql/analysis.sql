-- Multi-Touch Marketing Attribution Analysis
-- SQLite queries for customer journeys and attribution results.

.headers on
.mode column


-- ============================================================
-- 1. Overall marketing performance
-- ============================================================

SELECT
    COUNT(*) AS total_journeys,
    SUM(converted) AS conversions,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS conversion_rate_pct,
    ROUND(
        SUM(conversion_value),
        2
    ) AS total_revenue,
    ROUND(
        AVG(touchpoint_count),
        2
    ) AS average_touchpoints
FROM journey_summary;


-- ============================================================
-- 2. Compare converted and non-converted journeys
-- ============================================================

SELECT
    CASE
        WHEN converted = 1 THEN 'Converted'
        ELSE 'Not Converted'
    END AS journey_status,
    COUNT(*) AS journeys,
    ROUND(
        AVG(touchpoint_count),
        2
    ) AS average_touchpoints,
    ROUND(
        AVG(unique_channels),
        2
    ) AS average_unique_channels,
    ROUND(
        AVG(journey_duration_hours),
        2
    ) AS average_duration_hours,
    ROUND(
        AVG(conversion_value),
        2
    ) AS average_revenue
FROM journey_summary
GROUP BY converted
ORDER BY converted DESC;


-- ============================================================
-- 3. Highest-performing channel under each model
-- ============================================================

WITH ranked_channels AS (
    SELECT
        model,
        channel,
        attributed_conversions,
        attributed_revenue,
        revenue_share,
        ROW_NUMBER() OVER (
            PARTITION BY model
            ORDER BY attributed_revenue DESC
        ) AS revenue_position
    FROM attribution_channel_summary
)

SELECT
    model,
    channel AS top_channel,
    ROUND(
        attributed_conversions,
        2
    ) AS attributed_conversions,
    ROUND(
        attributed_revenue,
        2
    ) AS attributed_revenue,
    ROUND(
        revenue_share * 100,
        2
    ) AS revenue_share_pct
FROM ranked_channels
WHERE revenue_position = 1
ORDER BY model;


-- ============================================================
-- 4. Full channel ranking by attribution model
-- ============================================================

SELECT
    model,
    channel,
    channel_rank,
    ROUND(
        attributed_conversions,
        2
    ) AS attributed_conversions,
    ROUND(
        attributed_revenue,
        2
    ) AS attributed_revenue,
    ROUND(
        conversion_share * 100,
        2
    ) AS conversion_share_pct,
    ROUND(
        revenue_share * 100,
        2
    ) AS revenue_share_pct
FROM attribution_channel_summary
ORDER BY
    model,
    channel_rank,
    channel;


-- ============================================================
-- 5. Conversion rate for journeys touching each channel
-- ============================================================

WITH journey_channel AS (
    SELECT
        journey_id,
        channel,
        MAX(converted) AS converted
    FROM clean_customer_journeys
    GROUP BY
        journey_id,
        channel
)

SELECT
    channel,
    COUNT(*) AS journeys_touching_channel,
    SUM(converted) AS converted_journeys,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS assisted_conversion_rate_pct
FROM journey_channel
GROUP BY channel
ORDER BY
    assisted_conversion_rate_pct DESC,
    converted_journeys DESC;


-- ============================================================
-- 6. Most common customer journey paths
-- ============================================================

SELECT
    journey_path,
    COUNT(*) AS journeys,
    SUM(converted) AS conversions,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS conversion_rate_pct,
    ROUND(
        SUM(conversion_value),
        2
    ) AS revenue
FROM journey_summary
GROUP BY journey_path
HAVING COUNT(*) >= 5
ORDER BY
    conversions DESC,
    revenue DESC,
    journeys DESC
LIMIT 15;


-- ============================================================
-- 7. First-touch versus last-touch channel performance
-- ============================================================

SELECT
    first_channel AS channel,
    COUNT(*) AS first_touch_journeys,
    SUM(converted) AS first_touch_conversions,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS first_touch_conversion_rate_pct
FROM journey_summary
GROUP BY first_channel
ORDER BY first_touch_conversion_rate_pct DESC;


SELECT
    last_channel AS channel,
    COUNT(*) AS last_touch_journeys,
    SUM(converted) AS last_touch_conversions,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS last_touch_conversion_rate_pct
FROM journey_summary
GROUP BY last_channel
ORDER BY last_touch_conversion_rate_pct DESC;


-- ============================================================
-- 8. Device exposure and conversion performance
-- ============================================================

WITH journey_device AS (
    SELECT
        journey_id,
        device,
        MAX(converted) AS converted
    FROM clean_customer_journeys
    GROUP BY
        journey_id,
        device
)

SELECT
    device,
    COUNT(*) AS journey_device_exposures,
    SUM(converted) AS converted_exposures,
    ROUND(
        100.0 * SUM(converted) / COUNT(*),
        2
    ) AS conversion_rate_pct
FROM journey_device
GROUP BY device
ORDER BY conversion_rate_pct DESC;


-- ============================================================
-- 9. Campaign performance averaged across all models
-- ============================================================

SELECT
    channel,
    campaign,
    ROUND(
        SUM(attributed_conversions)
        / COUNT(DISTINCT model),
        2
    ) AS average_attributed_conversions,
    ROUND(
        SUM(attributed_revenue)
        / COUNT(DISTINCT model),
        2
    ) AS average_attributed_revenue
FROM attribution_touchpoint_credits
GROUP BY
    channel,
    campaign
ORDER BY average_attributed_revenue DESC
LIMIT 20;


-- ============================================================
-- 10. Monthly conversion and revenue trend
-- ============================================================

SELECT
    SUBSTR(
        conversion_timestamp,
        1,
        7
    ) AS conversion_month,
    COUNT(*) AS conversions,
    ROUND(
        SUM(conversion_value),
        2
    ) AS revenue,
    ROUND(
        AVG(conversion_value),
        2
    ) AS average_order_value
FROM journey_summary
WHERE converted = 1
GROUP BY conversion_month
ORDER BY conversion_month;