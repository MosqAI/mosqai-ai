-- Run against the mosqai_local database:
--   psql -U postgres -d mosqai_local -f db/schema.sql
CREATE TABLE IF NOT EXISTS trap_analytics_log (
    id SERIAL PRIMARY KEY,
    trap_id VARCHAR(50) NOT NULL,
    image_uid VARCHAR(255),
    new_mosquitoes_count INT NOT NULL,
    total_mosquitoes_count INT NOT NULL,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trap_time ON trap_analytics_log (trap_id, logged_at DESC);
