# AGENTS.md

AI-agent knowledge base for the tokscale project.

## Agent Command Execution

- When running `tokscale` CLI commands from an automated agent (tests, CI, or tool-driven shells), always pass `--no-spinner` unless spinner behavior is the thing being tested.
- This avoids non-interactive terminal issues and keeps command output stable for assertions and logs.

## Release & Deployment

### Overview

Releases are published to npm via a GitHub Actions `workflow_dispatch` pipeline, followed by a manually created GitHub Release with handwritten notes. There is no staging environment — publishes go directly to npm `latest`.

### Release Pipeline

**Workflow:** `.github/workflows/publish-cli.yml`

**Trigger:** Manual — GitHub Actions UI → "Publish" → "Run workflow"

**Inputs:**
- `bump`: Version bump type — `patch (x.x.X)` | `minor (x.X.0)` | `major (X.0.0)`
- `version` (optional): Override string (e.g., `2.0.0-beta.1`), takes precedence over bump

**Stages (sequential):**

| # | Job | Description |
|---|-----|-------------|
| 1 | `bump-versions` | Reads current version from `packages/cli/package.json`, calculates new version, updates all platform package.json files + CLI + wrapper, uploads as artifact |
| 2 | `build-cli-binary` | 8-target parallel native Rust builds (macOS x86/arm64, Linux glibc/musl x86/arm64, Windows x86/arm64) |
| 3 | `publish-platform-packages` | Publishes platform-specific packages (`@tokscale/cli-darwin-arm64`, etc.) containing native binaries to npm |
| 4 | `publish-cli` | Publishes `@tokscale/cli` to npm (binary dispatcher + optionalDependencies) |
| 5 | `publish-alias` | Publishes `tokscale` wrapper package to npm |
| 6 | `finalize` | Commits bumped `package.json` files back to repo as `chore: bump version to X.Y.Z` (authored by `github-actions[bot]`) |

**Duration:** ~15-20 minutes end-to-end.

**Package publish chain:** `@tokscale/cli` (with platform packages as optionalDependencies) → `tokscale` (depends on cli). Each waits for the previous to succeed.

### Post-Pipeline: Git Tag & GitHub Release

The CI pipeline does **NOT** create the git tag or GitHub Release. After the workflow completes successfully:

1. Verify the `chore: bump version to X.Y.Z` commit was pushed by CI
2. Create a GitHub Release manually:
   - **Tag:** `vX.Y.Z` (e.g., `v1.2.1`)
   - **Target:** The `chore: bump version to X.Y.Z` commit
   - **Title:** See [Release Notes Style](#release-notes-style) below
   - **Body:** See [Release Notes Template](#release-notes-template) below
3. Publish the release (not as draft, not as prerelease)

### Versioning Conventions

| Bump Type | When to Use | Example |
|-----------|-------------|---------|
| `patch` | Bug fixes, small features, additive parser support | `1.2.0` → `1.2.1` |
| `minor` | New client support, significant features, UI overhauls | `1.1.2` → `1.2.0` |
| `major` | Breaking changes (never used so far) | `1.2.1` → `2.0.0` |

Version is stored in 3 places (all updated by CI):
- `packages/cli/package.json` — source of truth
- Platform packages (`packages/cli-*/package.json`) — version synced
- `packages/tokscale/package.json` — version + `@tokscale/cli` dependency version

### CI-Only Workflow

**`.github/workflows/build-native.yml`** — Runs on PRs touching `crates/tokscale-cli/**`. Builds all 8 native targets to verify compilation. Does not publish.

---

### Release Notes Style

#### Title Conventions

| Release Type | Title Format |
|-------------|--------------|
| Standard patch/minor | `` `tokscale@vX.Y.Z` is here! `` |
| Flagship feature | `` EMOJI `tokscale@vX.Y.Z` is here! (Short subtitle with [link](...)) `` |
| Feature spotlight | Custom banner image replacing the standard hero + call-to-action |

**Examples from past releases:**
- Standard: `` `tokscale@v1.1.2` is here! ``
- Flagship: `` 🦞 `tokscale@v1.2.0` is here! (Now supports [OpenClaw](https://github.com/openclaw/openclaw)) ``
- Spotlight: Custom Wrapped 2025 banner + `` Generate your Wrapped 2025 with `tokscale@v1.0.16` ``

#### Release Notes Template

```markdown
<div align="center">

[![Tokscale](https://github.com/junhoyeo/tokscale/raw/main/.github/assets/hero-v2.png)](https://github.com/junhoyeo/tokscale)

# `tokscale@vX.Y.Z` is here!
</div>

## What's Changed
* scope(area): description by @author in https://github.com/junhoyeo/tokscale/pull/NNN
* scope(area): description by @author in https://github.com/junhoyeo/tokscale/pull/NNN

## New Contributors
* @username made their first contribution in https://github.com/junhoyeo/tokscale/pull/NNN

**Full Changelog**: https://github.com/junhoyeo/tokscale/compare/vPREVIOUS...vNEW
```

#### Style Rules

| Element | Rule |
|---------|------|
| **Header** | Always centered `<div align="center">` with hero banner image linked to the repo |
| **Title** | Backtick-wrapped `tokscale@vX.Y.Z` — package name, not just version |
| **PR list** | `* scope(area): description by @author in URL` — mirrors the PR title exactly as merged |
| **Optional summary** | For releases with many changes or when PR titles alone don't convey impact, add a brief bullet list between the title and "What's Changed" (see v1.0.18 as example) |
| **New Contributors** | Include section when there are first-time contributors |
| **Full Changelog** | Always present at bottom as a GitHub compare link `vPREV...vNEW` |
| **Tone** | Concise. No prose paragraphs. Let the PR list speak for itself. |
| **No draft issues** | Never reference draft release issues (e.g., #121) in the notes |

#### When to Add a Summary Block

Add a short bullet list summary (before "What's Changed") when:
- The release has 4+ PRs spanning different areas
- PR titles alone don't convey the user-facing impact
- A new client/integration is the headline

**Example (v1.0.18):**
```markdown
- Improved model price resolver (Rust)
- Add support for Amp (AmpCode) and Droid (Factory Droid)
- Improved sorting feature on TUI
```

### Deployment Checklist

```
1. [ ] All target PRs merged to main
2. [ ] `cargo test` passes in crates/tokscale-cli
3. [ ] No open blocker bugs (regressions from changes being released)
4. [ ] Run "Publish" workflow via GitHub Actions UI
   - Select bump type (patch/minor/major)
   - Wait for all 6 stages to complete
5. [ ] Verify `chore: bump version to X.Y.Z` commit was pushed
6. [ ] Verify packages on npm: @tokscale/cli, tokscale
7. [ ] Create GitHub Release
   - Tag: vX.Y.Z targeting the bump commit
   - Write release notes following the template above
   - Publish (not draft, not prerelease)
8. [ ] Smoke test: `bunx tokscale@latest --version`
```
