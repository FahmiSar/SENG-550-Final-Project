-- =====================================================================
-- SENG 550 Final Project: Smart Grid Electricity Consumption Forecasting
-- Star schema: fact_load (center) + dim_date + dim_meter
-- =====================================================================


DROP TABLE IF EXISTS fact_load;
DROP TABLE IF EXISTS dim_meter;
DROP TABLE IF EXISTS dim_date;


-- ---------------------------------------------------------------------
-- dim_date: one row per DAY
-- ---------------------------------------------------------------------
CREATE TABLE dim_date (
    date          DATE      PRIMARY KEY,                       
    weekday       SMALLINT  NOT NULL CHECK (weekday BETWEEN 0 AND 6),   -- 0 = Monday
    is_weekend    BOOLEAN   NOT NULL,
    month         SMALLINT  NOT NULL CHECK (month BETWEEN 1 AND 12),
    season        TEXT      NOT NULL CHECK (season IN ('dry', 'rainy', 'harmattan')),
    temperature   REAL,                                          
    is_rain_day   BOOLEAN   NOT NULL,
    holiday_type  TEXT      CHECK (holiday_type IN ('national', 'religious'))  -- NULL = not a holiday
);


-- ---------------------------------------------------------------------
-- dim_meter: one row per METER 
-- ---------------------------------------------------------------------
CREATE TABLE dim_meter (
    meter_id       TEXT  PRIMARY KEY,                            
    consumer_type  TEXT  NOT NULL CHECK (consumer_type IN ('residential', 'commercial', 'industrial')),
    load_share     REAL  NOT NULL CHECK (load_share > 0 AND load_share <= 1)  
);


-- ---------------------------------------------------------------------
-- fact_load: one row per HOUR per METER 
-- ---------------------------------------------------------------------
CREATE TABLE fact_load (
    meter_id   TEXT       NOT NULL,
    timestamp  TIMESTAMP  NOT NULL,
    date       DATE       NOT NULL,                             
    hour       SMALLINT   NOT NULL CHECK (hour BETWEEN 0 AND 23),
    load_kw    REAL       NOT NULL CHECK (load_kw >= 0),

    PRIMARY KEY (meter_id, timestamp),                           
    FOREIGN KEY (meter_id) REFERENCES dim_meter (meter_id),
    FOREIGN KEY (date)     REFERENCES dim_date (date)
);