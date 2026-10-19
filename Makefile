# Requires Docker (Docker Desktop with WSL2 integration on Windows).
RUN = docker compose run --rm runner
DASHBOARD_ENV = ANALYTICS_DATABASE_URL=postgresql://rad:rad@host.docker.internal:5432/rad \
                DASHBOARD_EVENTS_URL=http://host.docker.internal:5000/events

.PHONY: up up-fake up-integrated seed down logs ps test test-crash eval bench-load bench-llm clean

up:              ## full stack with DeepSeek (needs DEEPSEEK_API_KEY in .env), demo data seeded
	docker compose up -d --build
	$(RUN) python -m agentplat.seed
up-fake:         ## same stack, scripted model instead of DeepSeek (no API key, used by bench-load)
	LLM_PROVIDER=fake docker compose up -d --build
	$(RUN) python -m agentplat.seed
up-integrated:   ## tools read the analytics dashboard's database and the agent reports to its API
	$(DASHBOARD_ENV) docker compose up -d --build
seed:            ## refresh the standalone demo data (relative to now)
	$(RUN) python -m agentplat.seed
down:
	docker compose down
logs:
	docker compose logs -f agent mcp-server
ps:
	docker compose ps

test:            ## unit + integration tests (Postgres and SpiceDB from the stack, scratch database)
	$(RUN) python -m pytest -q
test-crash:      ## the kill-and-resume test, verbose
	$(RUN) python -m pytest -v -s tests/test_crash_resume.py
eval:            ## run the evaluation tasks with DeepSeek; ARGS="--repeat 3" or ARGS="--only id1,id2"
	$(RUN) python -m evals.run $(ARGS)
bench-load:      ## 50 concurrent sessions over 10 WebSockets (run after make up-fake)
	$(RUN) python bench/ws_load.py --api http://agent:8000 --ws ws://agent:8000/ws $(ARGS)
bench-llm:       ## DeepSeek time-to-first-token, without the platform
	$(RUN) python bench/llm_latency.py $(ARGS)
clean:           ## stop and delete all data
	docker compose down -v
