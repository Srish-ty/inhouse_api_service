CREATE TABLE IF NOT EXISTS sessions (
  session_id VARCHAR(64) PRIMARY KEY,
  app_name VARCHAR(128) NOT NULL,
  user_id VARCHAR(128) NOT NULL,
  state JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted BOOLEAN NOT NULL DEFAULT FALSE,
  metadata TEXT
);

CREATE TABLE IF NOT EXISTS session_events (
  event_id VARCHAR(64) PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  app_name VARCHAR(128) NOT NULL,
  user_id VARCHAR(128) NOT NULL,
  author VARCHAR(128) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  content_json JSONB,
  actions_json JSONB,
  metadata_json JSONB,
  error_code VARCHAR(64),
  error_message TEXT,
  invocation_id VARCHAR(128)
);

CREATE INDEX IF NOT EXISTS idx_session_events_session_time
  ON session_events (session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_sessions_app_user_last_update
  ON sessions (app_name, user_id, last_update_time DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_app_last_update
  ON sessions (app_name, last_update_time DESC);