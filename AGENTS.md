## Git Workflow

Every approved task or phase must end with a Git checkpoint.

Before committing:
1. Run the relevant tests or verification commands.
2. Confirm the Definition of Done is satisfied.
3. Inspect `git status` and `git diff`.
4. Never commit raw datasets, model weights, indexes, caches, secrets,
   virtual environments, or generated temporary files.
5. Never commit a failing or partially completed task unless explicitly requested.

When the task is complete:
1. Stage only files relevant to the task.
2. Create one meaningful commit using Conventional Commits.
3. Push the current branch to its configured remote.
4. Report:
   - commit hash
   - commit message
   - branch
   - push result

Commit examples:
- chore: establish project harness
- feat: analyze ViBioMIR dataset
- feat: implement resumable corpus crawler
- feat: add BGE-M3 dense retrieval
- test: add retrieval evaluator edge cases
- exp: compare chunking configurations

Do not create multiple commits for intermediate debugging attempts.
One completed logical task should normally produce one clean commit.

If push fails because no remote, authentication, or upstream is configured:
- do not invent credentials;
- preserve the local commit;
- report the exact Git issue;
- stop rather than changing repository configuration without approval.

## Experiment Record

An experiment is not complete until:

1. its result/lesson is appended to `docs/EXPERIMENT_JOURNAL.md`,
2. `docs/TIMELINE.md` is updated when it changes project direction,
3. `docs/leaderboard_history.csv` is updated for every organizer submission result,
4. claims are grounded in repository evidence; never reconstruct or invent
   missing history from memory,
5. the documentation changes above are included in the experiment's final Git
   checkpoint.

For a completed experiment, the required order is normally:

experiment
-> validation
-> journal/timeline update
-> git diff review
-> commit
-> push
-> report commit hash
