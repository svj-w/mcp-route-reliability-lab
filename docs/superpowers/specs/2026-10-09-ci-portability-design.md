# CI Portability Repair

The user confirmed this minimal repair after the first public push failed CI.
The public repository is svj-w/mcp-route-reliability-lab; starting commit is
f80972f05abffc4dc33887ae3623cb11c1fef837.

## Evidence and Scope

- Python 3.12 installed NumPy 2.5.3, whose stubs use Python 3.12 syntax.
  The workflow inherited mypy's project default python_version=3.11 and failed
  while parsing that dependency. Python 3.11 passed both mypy and pytest.
- Candidate regeneration produced four CSV diffs, totaling 424 removed and
  424 added lines. All lines match after removing carriage returns. Git stores
  LF; csv.DictWriter defaults to CRLF even on Linux when newline="" is used.
- Do not change query text, labels, split assignment, raw AI annotations, frozen
  reports, dependency constraints, or recorded source commits.

## Approved Design

Pass --python-version ${{ matrix.python-version }} to the CI mypy command.
Keep the project's default Python 3.11 floor, and continue both existing
3.11/3.12 matrix jobs. Each environment checks its own dependency stubs;
the 3.11 job retains minimum-version coverage.

Add lineterminator="\n" to the four candidate CSV writers, preserving their
field ordering and existing UTF-8/UTF-8-BOM encodings. Leave all committed data
untouched. A regression rebuilds the candidate in an isolated temporary root
and compares every output against the existing snapshot with checkout newline
normalization. All four regenerated CSVs must contain LF, not CRLF.

## Validation and Publication

First demonstrate the two regressions fail. Then run focused tests, the entire
local suite, Ruff, mypy, production-profile validation, and an offline fixture
benchmark. Independently review the small diff. Verify git diff is empty for
data/ and reports/, commit only the repair and its documentation/tests, and
push without force. Require both GitHub CI matrix jobs to finish successfully.
Do not claim Python 3.12 was executed locally: this machine has Python 3.11.
