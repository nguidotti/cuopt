=====================
Release Notes
=====================

====================
Release Notes 25.10
====================

New Features (25.10)
--------------------

- New barrier method for solving LPs. Uses cuDSS for sparse Cholesky / LDT.
- Concurrent mode for LPs now uses PDLP, dual simplex, and barrier
- New PDLP solver mode Stable3.
- MIP presolve using Papilo (enabled by default). LP presolve using Papilo (optional).
- Parallel branch and bound on the CPU: multiple best-first search and diving threads

Breaking Changes (25.10)
------------------------

- New PDLP Solver mode Stable3 is the default


Improvements (25.10)
--------------------

- Add setting "CUOPT_BARRIER_DUAL_INITIAL_POINT" to change the dual initial point used by barrier
- CPUFJ for local search + simple rounding
- FP as a local search
- Sub-MIP recombiner and B&B global variable changes
- Implement GF(2) presolve reduction
- Implement node presolve
- CUDA 13/12.9 support
- Build and test with CUDA 13.0.0
- Add read/write MPS and relaxation to python API
- Decompression for ``.mps.gz`` and ``.mps.bz2`` files
- Enable parallelism for root node presolve
- Enable singleton stuffing and use Papilo default params
- Make infeasibility checks consistent between the main solver and presolver
- Add maximization support for root node presolve
- Performance improvement in dual simplex's right-looking LU factorization
- Fix high GPU memory usage
- Print cuOpt version / machine info before solving
- ``cuopt-server``: update dependencies (drop httpx, add psutil)
- Add nightly testing of cuOpt jump interface
- Compression tests are not run when compression is disabled
- Add sanitizer build option- Heuristic Improvements: balance between generation and improvement heuristics
- Loosen presolve tolerance and update timers to report cumulative presolve/solve time
- Warn in case a dependent library is not found in libcuopt load
- Combined variable bounds
- Add Commit Sha to container for reference
- use GCC 14, consolidate dependency groups, update pre-commit hooks
- Add support for nightly ``cuopt-examples`` notebook testing
- Reduce hard-coded version usage in repo
- Container to work on all different users including root
- Changes to download LP and MILP datasets, and also disable cvxpy testing for 3.10
- Faster engine compile time
- Fix pre-commit for trailing whitespace and end of file
- Merge update version and fix version format bugs
- This library now supports the QPS format, which is an extension of the standard MPS format for representing quadratic programming problems.


Bug Fixes (25.10)
-----------------

- Fix variables out of bounds caused by CPUFJ LP scratch thread
- Fix the maybe-uninitialized compilation error
- Fix linking errors in the test suite when disabling C adaptor
- Compute relative gap with respect to user objectives
- Add http timeout values for general, send, and receive to client
- Fix bug in ``fixed_problem_computation``
- Remove ``limiting_resource_adaptor`` leftover
- Add support for cuda13 container and fix cuda13 lib issues in wheel
- Return Infeasible if the user problem contains crossing bounds
- Fix out-of-bound access in ``clean_up_infeasibilities``
- Empty columns with infinite bounds are not removed


Documentation (25.10)
---------------------

- Add tutorial video links to Decompression
- Add warmstart, model update, update docs
- add docs on CI workflow inputs
- Add name to drop-down for video link
- Add video link to the docs and to the Readme
- Add documentation on nightly installation commands
- Fix version in version tab, change log, and fix typos
- Doc update for container version update, and add ``nvidia-cuda-runtime`` as a dependency


====================
Release Notes 25.08
====================


New Features (25.08)
--------------------

- Added Python API for LP and MILP (`#223 <https://github.com/NVIDIA/cuopt/pull/223>`_)

Breaking Changes (25.08)
------------------------

- Fixed versioning for nightly and release package (`#175 <https://github.com/NVIDIA/cuopt/pull/175>`_)

Improvements (25.08)
--------------------

- New heuristic improvements (`#178 <https://github.com/NVIDIA/cuopt/pull/178>`_)
- Add helm chart for cuOpt service (`#224 <https://github.com/NVIDIA/cuopt/pull/224>`_)
- Add nightly container support (`#180 <https://github.com/NVIDIA/cuopt/pull/180>`_)
- Adding deb package support as a beta feature (`#190 <https://github.com/NVIDIA/cuopt/pull/190>`_)
- Use cusparsespmv_preprocess() now that Raft implements it (`#120 <https://github.com/NVIDIA/cuopt/pull/120>`_)
- Create a bash script to run MPS files in parallel (`#87 <https://github.com/NVIDIA/cuopt/pull/87>`_)
- Several fixes needed to compile cuOpt with LLVM (`#121 <https://github.com/NVIDIA/cuopt/pull/121>`_)
- Small fixes for corner cases (`#130 <https://github.com/NVIDIA/cuopt/pull/130>`_)
- Small improvements on how paths are handled in tests (`#129 <https://github.com/NVIDIA/cuopt/pull/129>`_)
- Update cxxopts to v3.3.1 (`#128 <https://github.com/NVIDIA/cuopt/pull/128>`_)
- Bump actions/checkout in nightly.yaml to v4 (`#230 <https://github.com/NVIDIA/cuopt/pull/230>`_)
- Remove CUDA 11 specific changes from repo (`#222 <https://github.com/NVIDIA/cuopt/pull/222>`_)
- Heuristic improvements with solution hash, MAB and simplex root solution (`#216 <https://github.com/NVIDIA/cuopt/pull/216>`_)
- Various typos in comments and strings, note on result dir (`#200 <https://github.com/NVIDIA/cuopt/pull/200>`_)
- Split very large tests into smaller individual test cases (`#152 <https://github.com/NVIDIA/cuopt/pull/152>`_)
- Fix compile error when using clang with C++20 (`#145 <https://github.com/NVIDIA/cuopt/pull/145>`_)
- Relax pinnings on several dependencies, remove nvidia channel (`#125 <https://github.com/NVIDIA/cuopt/pull/125>`_)
- Fix compile error when building with clang (`#119 <https://github.com/NVIDIA/cuopt/pull/119>`_)
- cuOpt service add healthcheck for / (`#114 <https://github.com/NVIDIA/cuopt/pull/114>`_)
- refactor(shellcheck): fix all remaining shellcheck errors/warnings (`#99 <https://github.com/NVIDIA/cuopt/pull/99>`_)
- Add CTK 12.9 fatbin flags to maintain existing binary sizes (`#58 <https://github.com/NVIDIA/cuopt/pull/58>`_)

Bug Fixes (25.08)
-----------------

- Fixed a segfault on bnatt500 due to small mu leading to inf/nan (`#254 <https://github.com/NVIDIA/cuopt/pull/254>`_)
- Fixed a bug in basis repair. Recover from numerical issues in primal update (`#249 <https://github.com/NVIDIA/cuopt/pull/249>`_)
- Unset NDEBUG in cmake in assert mode (`#248 <https://github.com/NVIDIA/cuopt/pull/248>`_)
- Manual cuda graph creation in load balanced bounds presolve (`#242 <https://github.com/NVIDIA/cuopt/pull/242>`_)
- Fixed bug on initial solution size in the check and cuda set device order (`#226 <https://github.com/NVIDIA/cuopt/pull/226>`_)
- Disable cuda graph in batched PDLP (`#225 <https://github.com/NVIDIA/cuopt/pull/225>`_)
- Fix logging levels format with timestamps (`#201 <https://github.com/NVIDIA/cuopt/pull/201>`_)
- Fix bug in scaling of dual slacks and sign of dual variables for >= constraints (`#191 <https://github.com/NVIDIA/cuopt/pull/191>`_)
- Fix inversion crossover bug with PDP and prize collection (`#179 <https://github.com/NVIDIA/cuopt/pull/179>`_)
- Fix a bug in extract_best_per_route kernel (`#156 <https://github.com/NVIDIA/cuopt/pull/156>`_)
- Fix several bugs appeared in unit testing of JuMP interface (`#149 <https://github.com/NVIDIA/cuopt/pull/149>`_)
- Fix incorrect reported solving time (`#131 <https://github.com/NVIDIA/cuopt/pull/131>`_)
- Fix max offset (`#113 <https://github.com/NVIDIA/cuopt/pull/113>`_)
- Fix batch graph capture issue caused by pinned memory allocator (`#110 <https://github.com/NVIDIA/cuopt/pull/110>`_)
- Fix bug in optimization_problem_solution_t::copy_from (`#109 <https://github.com/NVIDIA/cuopt/pull/109>`_)
- Fix issue when problem has an empty problem in PDLP (`#107 <https://github.com/NVIDIA/cuopt/pull/107>`_)
- Fix crash on models with variables but no constraints (`#105 <https://github.com/NVIDIA/cuopt/pull/105>`_)
- Fix inversion of constraint bounds in conditional bounds presolve (`#75 <https://github.com/NVIDIA/cuopt/pull/75>`_)
- Fix data initialization in create depot node for max travel time feature (`#74 <https://github.com/NVIDIA/cuopt/pull/74>`_)

Documentation (25.08)
---------------------

- Added more pre-commit checks to ensure coding standards (`#213 <https://github.com/NVIDIA/cuopt/pull/213>`_)
- Mention GAMS and GAMSPy in third-party modeling languages page in documentation (`#206 <https://github.com/NVIDIA/cuopt/pull/206>`_)
- Enable doc build workflow and build script for PR and Nightly (`#203 <https://github.com/NVIDIA/cuopt/pull/203>`_)
- Fix the link to Python docs in README (`#118 <https://github.com/NVIDIA/cuopt/pull/118>`_)
- Add link checker for doc build and test (`#229 <https://github.com/NVIDIA/cuopt/pull/229>`_)

====================
Release Notes 25.05
====================

New Features (25.05)
--------------------

- Added concurrent mode that runs PDLP and Dual Simplex together
- Added crossover from PDLP to Dual Simplex
- Added a C API for LP and MILP
- PDLP: Faster iterations and new more robust default PDLPSolverMode Stable2
- Added support for writing out mps file containing user problem. Useful for debugging

Breaking Changes (25.05)
------------------------

- NoTermination is now a NumericalError
- Split cuOpt as libcuopt and cuopt wheel

Improvements (25.05)
--------------------

- Hook up MILP Gap parameters and add info about number of nodes explored and simplex iterations
- FJ bug fixes, tests and improvements
- Allow no time limit in MILP
- Refactor routing
- Probing cache optimization
- Diversity improvements for routing
- Enable more compile warnings and faster compile by bypassing rapids fetch
- Constraint prop based on load balanced bounds update
- Logger file handling and bug fixes on MILP
- Add shellcheck to pre-commit and fix warnings

Bug Fixes (25.05)
-----------------

- In the solution, ``termination_status`` should be cast to correct enum.
- Fixed a bug using vehicle IDs in construct feasible solution algorithm.
- FP recombiner probing bug fix.
- Fix concurrent LP crashes.
- Fix print relative dual residual.
- Handle empty problems gracefully.
- Improve breaks to allow dimensions at arbitrary places in the route.
- Free var elimination with a substitute variable for each free variable.
- Fixed race condition when resetting vehicle IDs in heterogenous mode.
- cuOpt self-hosted client, some MILPs do not have all fields in ``lp_stats``.
- Fixed RAPIDS logger usage.
- Handle LP state more cleanly, per solution.
- Fixed routing solver intermittent failures.
- Gracefully exit when the problem is infeasible after presolve.
- Fixed bug on dual resizing.
- Fix occasional incorrect solution bound on maximization problems
- Fix inversion of constraint bounds in conditional bounds presolve
- Pdlp fix batch cuda graph
- Fix obj constant on max. Fix undefined memory access at root
- Allow long client version in service version check, this fixes the issue in case version is of the format 25.05.00.dev0

Documentation (25.05)
---------------------
- Restructure documementation to accomdate new APIs
