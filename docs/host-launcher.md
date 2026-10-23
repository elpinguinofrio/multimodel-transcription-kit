# Shared host launcher

The root `multimodel-transcription-kit` executable is independent of `.venv`, its absolute
shebang, and the VM's Python installation. It runs the same packaged core with a
copy of the already installed standalone CPython runtime and dependencies under
`.private/portable-macos-arm64-kit`. It does not download software or modify either
machine's PATH, home configuration, services or skill registrations.

Run it from this machine's view of the shared checkout:

```sh
./multimodel-transcription-kit --portable-info
./multimodel-transcription-kit --help
./multimodel-transcription-kit --list-models
./multimodel-transcription-kit recording.m4a -d existing-output-directory
```

For example, a checkout can appear at
`$HOME/VM-shared/projects/multimodel-transcription-kit` on the host and
`/Volumes/VM-shared/projects/multimodel-transcription-kit` in the VM. The launcher resolves
its own location, so it does not put either spelling into the interpreter path.
Host execution still needs a real host-side check; a relocated VM test is not that check.

The private bundle contains a wheel of this checkout, the existing CPython 3.14
runtime, and the existing requests, python-dotenv, websockets and transitive
dependencies. Their exact versions and wheel hash are in its manifest. These are
deployment copies, not a new environment downloaded from package indexes. A
fresh clone does not include the private bundle. Build it from an approved
installed environment with `scripts/build_portable.py --help`; a rebuild must use
a new destination and preserve the earlier bundle until cleanup is authorized.

The builder takes an explicit configuration and an explicit existing env file
under the shared root. The bundle keeps only that env file's relative path, not
its credentials. On startup the launcher materializes a new private config using
the current mount spelling, preserves the configured models/prompts/routes, and
uses `.private/portable-state/<machine>/` for its request state and usage history.
The default enhanced-audio cache is inside that state directory; temporary audio
stays in `.private/portable-tmp/`. Neither needs the consuming machine's home folder.
An explicit `--config` takes precedence and is not rewritten. Standard macOS
Homebrew executable directories are added only to this child process's PATH.

The portable deployment targets the host, where this shared root is a native
local directory. SQLite state must not be used over an SMB mount by concurrent
processes. The VM should retain its existing native installation and local
queue configuration. Host and VM queues do not coordinate their quotas; the
provider still enforces the account quota. Do not reset or replace an existing
queue to evade a known quota stop.

`--portable-info` reports the package/interpreter paths, shared env-file presence,
and ffmpeg/ffprobe discovery without printing secrets or contacting a provider.
No transcription model is executed by `--help`. For MP4/MOV extraction and long
recordings, ffmpeg and ffprobe must exist on the consuming machine; they are not
bundled or installed by this launcher. A missing tool remains an explicit blocker.

Validation must include: missing VM venv and minimal PATH, a different mount
spelling and empty home, secrets/config path resolution, ordinary package tests,
and an actual host invocation. Report the last item separately when host command
execution is unavailable. Keep all build and test artifacts under the shared root.
