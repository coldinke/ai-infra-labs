# Remote GPU Workflow

This repo assumes local editing plus remote GPU execution. The remote machine
does not need git credentials; `scripts/gpu_remote_sync.sh` syncs the local
worktree over SSH and pulls profiler outputs back.

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
GPU_HOST=root@example.com
GPU_PORT=22
GPU_REMOTE_DIR=/root/ai-infra-labs
GPU_LAB=all
```

For fish shell, either rely on `.env` through `just` or export variables
directly:

```fish
set -gx GPU_HOST root@example.com
set -gx GPU_PORT 52882
set -gx GPU_REMOTE_DIR /root/ai-infra-labs
set -gx GPU_LAB all
```

Use an absolute `GPU_REMOTE_DIR` when the remote target is a container or cloud
workspace. Avoid `~` because it is expanded by different shells at different
times.

## Common Commands

```bash
just check-remote
just push
just remote pwd
just pull
just pull-raw
```

`just push` copies the current local worktree to the remote directory. It
excludes `.git`, Python build output, compiled objects, raw profiler files, and
`labs/*/profiles/`.

`just pull` copies `labs/*/profiles/` back from the remote by default, excluding
large raw profiler files such as `.nsys-rep`, `.sqlite`, and `.ncu-rep`.

Use `just pull-raw` only when a local GUI or detailed offline inspection needs
the raw profiler database.

## Running Labs Remotely

Generic remote execution:

```bash
just remote-test 000-cuda-rmsnorm
just remote-bench 000-cuda-rmsnorm rmsnorm --variant all
just remote-profile 000-cuda-rmsnorm rmsnorm --variant scalar --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10
```

Current RMSNorm shortcuts:

```bash
just rmsnorm-build
just rmsnorm-test
just rmsnorm-bench
just rmsnorm-profile
```

## Sync Semantics

- Push is incremental through `rsync`.
- Push deletes remote files that no longer exist locally, except for excluded
  paths. Use `--no-delete` with `scripts/gpu_remote_sync.sh` if needed.
- Pull is incremental through `rsync`.
- Pull defaults to all labs so switching to the next lab does not require a new
  sync command.
- The remote copy has no `.git` directory by design. `env.md` may show an
  unknown branch or commit on the remote.
