-- Capa CLEAN (datos normalizados). RAW vive en data/raw (parquet/json + manifest); FEATURES en schema `features`.
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS features;

CREATE TABLE IF NOT EXISTS clean.circuits (
  circuit_id VARCHAR PRIMARY KEY, name VARCHAR, country VARCHAR, locality VARCHAR,
  lat DOUBLE, lon DOUBLE,
  length_km DOUBLE, n_corners INTEGER, characteristics VARCHAR   -- metadatos curados a mano (fase 2)
);
CREATE TABLE IF NOT EXISTS clean.races (
  race_id VARCHAR PRIMARY KEY, season INTEGER, round INTEGER, name VARCHAR,
  circuit_id VARCHAR, race_date DATE
);
CREATE TABLE IF NOT EXISTS clean.drivers (
  driver_id VARCHAR PRIMARY KEY, code VARCHAR, name VARCHAR, nationality VARCHAR, number INTEGER
);
CREATE TABLE IF NOT EXISTS clean.teams (team_id VARCHAR PRIMARY KEY, name VARCHAR);

CREATE TABLE IF NOT EXISTS clean.race_results (
  race_id VARCHAR, driver_id VARCHAR, team_id VARCHAR,
  grid INTEGER,                 -- 0 = salida desde pit lane (convención Ergast/Jolpica)
  finish_position INTEGER, position_text VARCHAR, classified BOOLEAN,
  points DOUBLE, status VARCHAR, laps_completed INTEGER,
  PRIMARY KEY (race_id, driver_id)
);
CREATE TABLE IF NOT EXISTS clean.qualifying_results (
  race_id VARCHAR, driver_id VARCHAR, team_id VARCHAR, position INTEGER,
  q1_s DOUBLE, q2_s DOUBLE, q3_s DOUBLE,
  PRIMARY KEY (race_id, driver_id)
);
CREATE TABLE IF NOT EXISTS clean.laps (
  race_id VARCHAR, driver_id VARCHAR, lap_number INTEGER, lap_time_s DOUBLE,
  sector1_s DOUBLE, sector2_s DOUBLE, sector3_s DOUBLE,
  compound VARCHAR, tyre_life INTEGER, stint INTEGER, position INTEGER,
  pit_in BOOLEAN, pit_out BOOLEAN, track_status VARCHAR, is_accurate BOOLEAN,
  PRIMARY KEY (race_id, driver_id, lap_number)
);
CREATE TABLE IF NOT EXISTS clean.pit_stops (
  race_id VARCHAR, driver_id VARCHAR, stop_number INTEGER, lap INTEGER, duration_s DOUBLE,
  compound_before VARCHAR, compound_after VARCHAR,
  PRIMARY KEY (race_id, driver_id, stop_number)
);
CREATE TABLE IF NOT EXISTS clean.weather (
  race_id VARCHAR, t_offset_s DOUBLE, air_temp DOUBLE, track_temp DOUBLE,
  humidity DOUBLE, pressure DOUBLE, wind_speed DOUBLE, rainfall BOOLEAN
);
CREATE TABLE IF NOT EXISTS clean.race_control (
  race_id VARCHAR, t_offset_s DOUBLE, lap INTEGER, category VARCHAR, message VARCHAR
);


-- Ritmo de carrera por piloto (derivado de clean.laps). POST_RACE: solo se usa desplazado, vía features prior_*.
CREATE TABLE IF NOT EXISTS features.race_pace (
  race_id VARCHAR, driver_id VARCHAR, race_pace_pct DOUBLE, n_clean_laps INTEGER, race_wet BOOLEAN,
  PRIMARY KEY (race_id, driver_id)
);
