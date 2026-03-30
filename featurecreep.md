# Multi-repo benchmark support

## Goal

Make search-bench run against multiple codebases of varying size so we can answer: "does RAG help more as repos get bigger?"

## Benchmark packs

Each target repo becomes a self-contained "benchmark pack" under `benchmarks/`:

```
benchmarks/
  kicad-library/
    manifest.yaml
    queries.json
    tasks.json
  django/
    manifest.yaml
    queries.json
    tasks.json
  react/
    manifest.yaml
    queries.json
    tasks.json
  linux-kernel/
    manifest.yaml
    queries.json
    tasks.json
```

### Manifest format

```yaml
name: react
repo: facebook/react
ref: v19.1.0               # pinned for reproducibility
subset: null                # or "kernel/" for linux
size: large                 # small / medium / large / mega
languages: [javascript, typescript]
description: "React UI library — reconciler, scheduler, hooks, fiber architecture"
```

### Target repos

| Tier | Repo | ~Files | Language | Rationale |
|------|------|--------|----------|-----------|
| Small | michaelayles/kicad-library | 200 | TS | Already working. Baseline where native should be fine. |
| Medium | django/django | 2k | Python | Well-understood framework. ORM, middleware, templates give good cross-cutting queries. |
| Large | facebook/react | 5k | JS/TS | Deep internals (reconciler, fiber, scheduler). Conceptual queries are hard without semantic search. |
| Mega | torvalds/linux (`kernel/` subset) | 1k-2k scoped | C | Scoped to `kernel/` to keep indexing sane. Scheduler, cgroups, namespaces — deeply intertwined. |

## Implementation plan

### Phase 1: Restructure existing code

1. Move `queries/queries.json` and `queries/smoke_queries.json` into `benchmarks/kicad-library/`
2. Move `tasks/tasks.json` and `tasks/smoke_tasks.json` into `benchmarks/kicad-library/`
3. Create `benchmarks/kicad-library/manifest.yaml`
4. Update `runner.py` to accept `--benchmark kicad-library` instead of `--codebase` + `--queries` + `--tasks`
5. Runner reads manifest, resolves paths:
   - Clone dir: `benchmark/{name}/repo/` (or use existing if present)
   - Index paths: `data/{name}.db`, `data/{name}.faiss`
   - Results: `results/{name}/`
6. Keep `--codebase`/`--queries`/`--tasks` as overrides for ad-hoc use

### Phase 2: Auto-clone and index

1. `search-bench setup <benchmark-name>` command:
   - Reads manifest
   - Clones repo at pinned ref (or pulls if exists)
   - If `subset` is set, sparse-checkout that path only
   - Runs indexer
2. `search-bench setup --all` to prep everything
3. Skip already-cloned/indexed repos (check ref matches)

### Phase 3: Query generation scaffolding

Writing 60 ground-truth queries per repo by hand is the bottleneck. Semi-automate it:

1. `search-bench generate-queries <benchmark-name>` command
2. Indexes the repo, then generates candidate queries from the index:
   - **exact_symbol**: pick top-N symbols from AST chunks, generate "where is X defined?"
   - **conceptual**: use LLM to read key files and generate "how does X work?" questions
   - **cross_cutting**: find files that share imports/symbols across directories
   - **refactoring**: identify large files or repeated patterns
3. Output is a draft `queries.json` with empty `ground_truth` fields
4. Human reviews, fills in ground truth, removes bad candidates
5. Smoke subset: first 5-6 queries auto-tagged for `smoke_queries.json`

### Phase 4: Multi-benchmark runner

1. `--benchmark all` runs every benchmark pack sequentially
2. `--benchmark kicad-library,react` runs a subset
3. Per-benchmark results stored separately in `results/{name}/`
4. Global concurrency still applies across benchmarks

### Phase 5: Cross-benchmark analysis

1. Aggregate results across benchmarks grouped by `size` tier
2. New analysis outputs:
   - "Native vs RAG recall by repo size" chart
   - "Tool ranking by repo size" table
   - Per-benchmark report + combined summary
3. Statistical tests compare native vs RAG **per size tier**, not just overall

## Things to not over-engineer

- Don't build a package registry for benchmark packs. They're just directories with JSON + YAML.
- Don't auto-generate ground truth without human review. Bad ground truth is worse than fewer queries.
- Don't try to support arbitrary repo structures. If a repo needs special handling (sparse checkout, submodules), put that in the manifest and handle it in setup.
- Don't parallelise across benchmarks. Run them sequentially. The tool semaphores handle within-benchmark concurrency.

## Estimated query counts per repo

| Repo | exact_symbol | conceptual | cross_cutting | refactoring | Total |
|------|-------------|------------|---------------|-------------|-------|
| kicad-library | 15 | 15 | 15 | 15 | 60 (already done) |
| django | 20 | 15 | 15 | 10 | 60 |
| react | 20 | 15 | 15 | 10 | 60 |
| linux-kernel | 15 | 15 | 20 | 10 | 60 |

Tasks (author phase): 10-20 per repo. These are harder to write and validate, so start with read-only queries and add tasks later.

## Order of work

1. Phase 1 (restructure) — mechanical, no new features
2. Write django queries — manual, time-consuming, do alongside phase 1
3. Phase 2 (auto-clone/index) — quality of life
4. Phase 3 (query scaffolding) — speeds up query writing for react + linux
5. Write react + linux queries — using scaffolding output
6. Phase 4 (multi-benchmark runner) — ties it together
7. Phase 5 (cross-benchmark analysis) — the payoff
