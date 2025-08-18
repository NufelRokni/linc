#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Robust submit helper for executing the JOB string.
# Behaviors:
# - Fail early if JOB is empty.
# - Run the JOB under "bash -lc" so complex commands and expansions behave as expected.
# - Forward SIGINT/SIGTERM to the child process so it can clean up.
# - Print start/finish diagnostics and exit with the JOB's exit code.

if [[ -z "${JOB:-}" ]]; then
	echo "SUBMIT.sh: ERROR: JOB is empty" >&2
	exit 2
fi

echo "SUBMIT.sh: launching job at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >&2

# Start the job in the background so we can trap and forward signals.
bash -lc "set -o pipefail; ${JOB}" &
child_pid=$!

trap 'rc=$?; echo "SUBMIT.sh: caught signal, forwarding to child ${child_pid}" >&2; kill -TERM "${child_pid}" 2>/dev/null || true; exit ${rc:-130}' INT TERM

# Wait for the job to finish and capture exit code.
wait ${child_pid}
rc=$?

echo "SUBMIT.sh: job finished with exit code ${rc} at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >&2

exit ${rc}
