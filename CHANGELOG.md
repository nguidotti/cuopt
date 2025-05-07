# cuOpt 23.02.00 (Date TBD)

Please see https://github.com/rapidsai/cuopt/releases/tag/v23.02.00a for the latest changes to this development branch.

# cuOpt 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Infeasible Local Search (#796) @akifcorduk
- GPU cycle graph, random cycle finder, increased parallism on local search, improvements. (#740) @akifcorduk

## 🐛 Bug Fixes

- Don't use CMake 3.25.0 due to a FindCUDAToolkit show stopping bug (#853) @robertmaynard
- Fix buffer creation which changed in cuDF (#829) @rgsl888prabhu
- Add more checks for conflicts between break time windows and vehicle time windows (#811) @rg20
- Improve checks on conflicts between break time windows and vehicle time windows (#805) @rg20
- Update raft and pylibraft version to 22.12 (#785) @rgsl888prabhu

## 🚀 New Features

- Patch install-cnc.sh to use ansible 6.0.0 (#871) @tmckayus
- Implement squeeze procedure that inserts a request to a least excess position (#852) @akifcorduk
- Add vehicle dependent service time (#832) @hlinsen
- Implement vehicle max travel time constraint similar to max cost per route constraint (#815) @hlinsen
- Implements new objective functions to minimize variance of route sizes and route service times (#810) @rg20
- Add best cycle finder (#739) @hlinsen

## 🛠️ Improvements

- update_server_version_and_server_data_validation_bug (#879) @Iroy30
- Add cloud script tests (#870) @tmckayus
- Added microservice performance testing (#865) @SchultzC
- Run DLI notebooks on CI (#856) @Iroy30
- Sync cloud-scripts with cuopt-resources/cloud-scripts (#850) @tmckayus
- Implement task ids and provide support for newly added features on server API (#844) @Iroy30
- Enable tests for ci or conda env file changes (#831) @rgsl888prabhu
- API extractor script (#830) @anandhkb
- Fix conda env discrepancy in cuda-python in cudf and cuOpt (#819) @rgsl888prabhu
- Cloud ci test (#812) @anandhkb
- Infeasible Local Search (#796) @akifcorduk
- GPU cycle graph, random cycle finder, increased parallism on local search, improvements. (#740) @akifcorduk

# cuOpt 22.10.00 (12 Oct 2022)

## 🚨 Breaking Changes

- Adding drop infeasible orders wrapper to python and server (#737) @rgsl888prabhu
- Add mixed fleet max cost per vehicle (#730) @hlinsen
- Remove the requirement that the first order location should be corresponding to depot (#702) @rg20
- add multi cost matrix support to cuopt_server (#692) @Iroy30
- Solver API refactoring for consistent naming and modularization (#673) @rgsl888prabhu

## 🐛 Bug Fixes

- Bug fixes to handle order and break times (#813) @rgsl888prabhu
- Fix Vanilla PDP filter bug (#798) @akifcorduk
- Fix a bug in finding the farthest node as a seed for insertion (#766) @rg20
- Initialize per thread objective values when objective functions are used (#748) @rg20
- Use 32 bit integers to represent demand and capacity to avoid overflowing (#725) @rg20
- Do not insert order nodes that have conflicting time windows with breaks to avoid infeasible solutions (#721) @rg20
- Fix merge issue (#719) @akifcorduk
- Branch 22.10 merge 22.08 (#716) @rgsl888prabhu
- Fix build script on PR_ID usage on branch builds which doesn't have PR_ID (#669) @rgsl888prabhu

## 📖 Documentation

- Update docs with nvidia docs review and API changes (#723) @rgsl888prabhu

## 🚀 New Features

- Adding drop infeasible orders wrapper to python and server (#737) @rgsl888prabhu
- Add mixed fleet max cost per vehicle (#730) @hlinsen
- Implement order priorities in the context of drop infeasible orders feature (#718) @rg20
- Adding UCF files used to create cuOpt and cuOpt test client microservices (#703) @rgsl888prabhu
- cuOpt dashboard (#701) @anandhkb
- Solver API refactoring for consistent naming and modularization (#673) @rgsl888prabhu

## 🛠️ Improvements

- Adds re-optimization server notebook on PDP problem (#797) @rgsl888prabhu
- Breaks issue with vehicle time windows (#777) @rg20
- add arrival stamp, Task definition, update server display (#738) @Iroy30
- Adds job priority support to the server (#729) @rgsl888prabhu
- Adds vehicle order match and order vehicle match support to server (#728) @rgsl888prabhu
- Test notebooks on CI (#720) @Iroy30
- Improvements on GES (#717) @akifcorduk
- add type casting warning (#715) @Iroy30
- Updating for upcoming raft changes to mdspan/mdarray/span includes (#708) @cjnolet
- dashboard improvements for next iter (#706) @anandhkb
- Remove the requirement that the first order location should be corresponding to depot (#702) @rg20
- Multi-thread/stream execution, thrust 11.7, cuopt 22.10 (#693) @akifcorduk
- add multi cost matrix support to cuopt_server (#692) @Iroy30
- Add python tests (#672) @Iroy30
- Add cache tool option to speedup local build (#666) @hlinsen
- Change run_server.sh to run cuOpt server in detached mode (22.10) (#657) @tmckayus
- Update container builder to run new server and other enhancements (#654) @rgsl888prabhu

# cuOpt 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Fix server on handling pick-up and delivery (#612) @rgsl888prabhu
- Run only relevant tests based on the ChangeList (#592) @anandhkb

## 🐛 Bug Fixes

- Fix handling of precedence constraints on server side (#711) @Iroy30
- Fix python tests for max lateness and max distance (#648) @rgsl888prabhu
- Fix CUOPT_CURRENT_MONTH in cmake (#629) @akifcorduk
- Reverting to kmeans_deprecated (#622) @cjnolet
- Fix server on handling pick-up and delivery (#612) @rgsl888prabhu
- Update dependencies to 22.08 and add version update script (#604) @hlinsen
- Fix drop infeasible orders option (#599) @rgsl888prabhu
- Fix race condition in insertion heuristic (#593) @akifcorduk
- Use order locations when break locations are set (#590) @hlinsen
- Fix notebook cost matrix image (#589) @SchultzC
- Fix file extensions in handling PDP test cases (#575) @akifcorduk

## 📖 Documentation

- Adding examples to docs and updating docs (#563) @rgsl888prabhu

## 🚀 New Features

- Add a notebook to demonstrate the usage of multi cost matrix for mixed fleet modeling (#627) @rg20
- Implement vehicle to order matching constraints (#609) @rg20
- Configfile save (#607) @anandhkb
- Adds multi depot, vehicle break and precedence constraint support to server (#588) @rgsl888prabhu
- Add support for vehicle types and multiple cost/constraint matrices (#569) @hlinsen
- Implement a script to model dynamic optimization in the context of pickup and delivery problem (#561) @rg20
- GES and Graph Cycle Based Local Search for PDP problems (#524) @akifcorduk

## 🛠️ Improvements

- Add 6 month expiration to unlimited version of cuOpt (#660) @rgsl888prabhu
- Use symmetric cost matrix in mixed fleet notebook (#649) @rg20
- Make vehicle order matching API consistent with other APIs (#635) @rg20
- Adding container environment set-up scripts and utilities for running container (#620) @rgsl888prabhu
- Feature and API testing in python (#616) @Iroy30
- Add logging of Constraint violations for infeasible solves (#613) @Iroy30
- Enable codecoverage for cuOpt Python (#611) @anandhkb
- Add an option to disable tabu search in SAT solver (#610) @akifcorduk
- Update style checks and add pre-commit support (#606) @rgsl888prabhu
- Adds sync endpoint to server and supports order locations (#603) @rgsl888prabhu
- Pin max version of cuda-python to 11.7.0 (#595) @rgsl888prabhu
- Run only relevant tests based on the ChangeList (#592) @anandhkb
- Adds max distance per route, objective function and skip first trip support to async server (#579) @rgsl888prabhu
- Add container builder (#574) @rgsl888prabhu
- Branch 22.08 merge 22.06 2 (#562) @rgsl888prabhu
- Branch 22.08 merge 22.06 (#527) @rgsl888prabhu

# cuOpt 22.06.00 (7 June 2022)

## 🚨 Breaking Changes

- Remove sparse implementation (#427) @rg20

## 🐛 Bug Fixes

- Fix a bug in waypoint_matrix functions to enable runtime checks (#512) @Kh4ster
- Add missing pyraft to conda env 11.6 (#488) @rgsl888prabhu
- Prevent precedence for MG (#482) @akifcorduk
- Add assignment test and fix bugs in SAT solver (#474) @akifcorduk
- Fix to allow call to compute_waypoint_sequence without prior set_order_locations call (#473) @Kh4ster
- Fix a bug in determining smallest route in each climber (#464) @rg20
- Fix demand and candidate overflow (#451) @hlinsen
- Fix demand fill and update user guess with recent features (#369) @hlinsen

## 📖 Documentation

- Remove version from rst, it will be added from config file (#442) @rgsl888prabhu

## 🚀 New Features

- Support multi-depot routing (#505) @rg20
- Support constraint on maximum distance allowed per route/vehicle (#485) @hlinsen
- Implement the option to exclude cost and time of first trip for specific vehicles in the fleet (#476) @rg20
- Add compute_shortest_path_costs function (custom weight set) to waypoint matrix (#470) @Kh4ster
- Enable clean for python in build file (#469) @rgsl888prabhu
- Precedence constraints (#466) @akifcorduk
- Adding limited version as a build option (#437) @rgsl888prabhu
- Remove sparse implementation (#427) @rg20
- update cuopt to support csr graph &amp; get path between target locations (#420) @Kh4ster
- Expose solver hyper parameters via environment variables (#494) @akifcorduk
- Enable dropping of orders in case of an infeasible solution for successful solve (#503) @Iroy30
- Add MG python stack and CVRPTW benchmark notebook (#507) @hlinsen
- Enable multi depot feature for limited version (#529) @hlinsen

## 🛠️ Improvements

- Enable generic integer and floating values to be able support high precision computation (#494) @akifcorduk
- Refactoring to handle drop return trip (mixed fleet option) feature implicitly in distance calculation (#484) @rg20
- RAFT RNG API: using new Raft RNG API and made includes explicit (#475) @MatthiasKohl
- Remove use_secondary_matrix as a template parameter to reduce the code complexity (#445) @rg20
- Update rapids cmake (#412) @robertmaynard
- Expanded Example Notebooks (#498)

# cuOpt 22.04.00 (13 Apr 2022)

## 🚨 Breaking Changes

- Update breaks api ([#407](407)) [@rg20](https://github.com/rg20)
- Rename to cuopt in C++, cmake, and build scripts, python, notebooks ([#391](391)) [@rg20](https://github.com/rg20)
- Rename ReOpt to cuOptima in doc, ngc scripts and helm charts ([#390](390)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)

## 🐛 Bug Fixes

- Fix out of bounds issue for pickup and delivery case in initial insertion ([#368](368)) [@rg20](https://github.com/rg20)
- Fix cython for multiple break dimensions ([#359](359)) [@hlinsen](https://github.com/hlinsen)
- Fix print function to show travel time for vehicles ([#350](350)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Cap the time windows to 32 bit max when they are not specified ([#343](343)) [@rg20](https://github.com/rg20)

## 📖 Documentation

- Stop nbsphynx from executing ipynb files and links to source code ([#406](406)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Rename ReOpt to cuOptima in doc, ngc scripts and helm charts ([#390](390)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Adding enums to docs ([#375](375)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Adds new document style and example ([#313](313)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)

## 🚀 New Features

- Update breaks api ([#407](407)) [@rg20](https://github.com/rg20)
- Revert the member zone name to reopt ([#402](402)) [@rg20](https://github.com/rg20)
- Change Reopt to cuOpt ([#401](401)) [@rg20](https://github.com/rg20)
- Rename everything except weblink ([#400](400)) [@rg20](https://github.com/rg20)
- Pickup and Delivery location will be identified in results ([#394](394)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Rename to cuopt in C++, cmake, and build scripts, python, notebooks ([#391](391)) [@rg20](https://github.com/rg20)
- Add utility functions to write data model to hdf and read from hdf ([#388](388)) [@rg20](https://github.com/rg20)
- Add pickup and delivery benchmark scripts ([#386](386)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Adding helm charts for ea and unrestricted ([#348](348)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Add vehicle breaks and model break locations ([#336](336)) [@hlinsen](https://github.com/hlinsen)
- Script to add new users and update markdown to ngc ([#331](331)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- WalkSAT POC ([#329](329)) [@akifcorduk](https://github.com/akifcorduk)

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality ([#408](408)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `.github/ops-bot.yaml` config file ([#392](392)) [@ajschmidt8](https://github.com/ajschmidt8)
- Handle infeasible solutions with breaks and re-optimization ([#380](380)) [@rg20](https://github.com/rg20)
- Upgrade thrust, raft, rmm, RNG and version ([#363](363)) [@akifcorduk](https://github.com/akifcorduk)
- Add soft tw and max lateness per route for pickup and delivery ([#356](356)) [@hlinsen](https://github.com/hlinsen)
- Add objective value getter to assignment class ([#349](349)) [@hlinsen](https://github.com/hlinsen)
- Add objective function to server and fix bugs ([#347](347)) [@rgsl888prabhu](https://github.com/rgsl888prabhu)
- Use compile time array access ([#341](341)) [@hlinsen](https://github.com/hlinsen)
