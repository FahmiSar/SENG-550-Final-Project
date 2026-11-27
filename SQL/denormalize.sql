-- =====================================================================
-- SENG 550 Final Project: Smart Grid Electricity Consumption Forecasting
-- Denormalised load table: fact_load + dim_date + dim_meter in one table
-- =====================================================================


DROP MATERIALIZED VIEW IF EXISTS mv_load_denorm;
DROP VIEW IF EXISTS v_load_denorm;


-- ---------------------------------------------------------------------
-- v_load_denorm: one row per HOUR per METER (joins run on every query)
-- ---------------------------------------------------------------------
CREATE VIEW v_load_denorm AS
SELECT
    f.meter_id,
    m.consumer_type,
    m.load_share,
    f.timestamp,
    f.date,
    f.hour,
    f.load_kw,
    d.weekday,
    d.is_weekend,
    d.month,
    d.season,
    d.temperature,
    d.is_rain_day,
    d.holiday_type
FROM fact_load f
JOIN dim_meter m ON m.meter_id = f.meter_id
JOIN dim_date  d ON d.date     = f.date;


-- ---------------------------------------------------------------------
-- mv_load_denorm: same rows, precomputed and indexed
-- Rebuilt every time etl.py reloads the tables
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW mv_load_denorm AS
SELECT * FROM v_load_denorm;

CREATE INDEX idx_mv_load_denorm_ts       ON mv_load_denorm (timestamp);
CREATE INDEX idx_mv_load_denorm_meter_ts ON mv_load_denorm (meter_id, timestamp);