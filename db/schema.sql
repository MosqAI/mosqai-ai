-- Run in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run; it also upgrades a table created by an earlier version of this file.
CREATE TABLE IF NOT EXISTS public.trap_analytics_log (
    id BIGSERIAL PRIMARY KEY,
    trap_id VARCHAR(64) NOT NULL,
    image_uid VARCHAR(255),
    new_mosquitoes_count INT NOT NULL DEFAULT 0,
    total_mosquitoes_count INT NOT NULL DEFAULT 0,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 'frame' rows are written for every image the trap sends (also when nothing new
-- was found, so a silent trap is distinguishable from an empty one); 'reset' rows
-- mark the chamber being emptied. detected_count is the raw model count for the frame.
ALTER TABLE public.trap_analytics_log
    ADD COLUMN IF NOT EXISTS event_type VARCHAR(16) NOT NULL DEFAULT 'frame',
    ADD COLUMN IF NOT EXISTS detected_count INT NOT NULL DEFAULT 0;

-- Upgrade rows from the old format, which marked resets with a magic image_uid.
UPDATE public.trap_analytics_log
    SET event_type = 'reset', image_uid = NULL
    WHERE image_uid = 'RESET_VECTOR';
UPDATE public.trap_analytics_log
    SET detected_count = total_mosquitoes_count
    WHERE event_type = 'frame' AND detected_count = 0 AND total_mosquitoes_count > 0;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'trap_analytics_log_event_type_check') THEN
        ALTER TABLE public.trap_analytics_log
            ADD CONSTRAINT trap_analytics_log_event_type_check CHECK (event_type IN ('frame', 'reset'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_trap_time ON public.trap_analytics_log (trap_id, logged_at DESC);

-- RLS on with no policies: only the secret / service_role key (SUPABASE_SECRET_KEY)
-- can read or write, which is what the AI service uses. Public clients get nothing.
ALTER TABLE public.trap_analytics_log ENABLE ROW LEVEL SECURITY;
