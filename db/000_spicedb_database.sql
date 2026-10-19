-- Runs once, when the Postgres volume is first created.
-- SpiceDB keeps its relationships in its own database; the test suite gets a scratch database so
-- `make test` never touches the demo data.
CREATE DATABASE spicedb;
CREATE DATABASE agent_test;
