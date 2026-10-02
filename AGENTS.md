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