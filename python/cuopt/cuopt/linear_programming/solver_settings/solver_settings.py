# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum, auto


class SolverMode(IntEnum):
    """
    Enum representing different solver modes to use in the
    `SolverSettings.set_pdlp_solver_mode` function.

    Attributes
    ----------
    Stable2
        Best overall mode from experiments; balances speed and convergence
        success. If you want to use the legacy version, use Stable1.
    Methodical1
        Takes slower individual steps, but fewer are needed to converge.
    Fast1
        Fastest mode, but with less success in convergence.

    Notes
    -----
    Default mode is Stable2.
    """

    Stable1 = 0
    Stable2 = auto()
    Methodical1 = auto()
    Fast1 = auto()
    DualSimplex = auto()


def toDict(model):

    if not isinstance(model, SolverSettings):
        raise ValueError("model must be a solver_settings.SolverSettings")

    solver_config = {
        "tolerances": {},
        "infeasibility_detection": model.detect_infeasibility,
        "time_limit": model.time_limit,
        "iteration_limit": model.iteration_limit,
        "solver_mode": model.solver_mode,
        "mip_scaling": model.mip_scaling,
        "heuristics_only": model.mip_heuristics_only,
        "num_cpu_threads": model.mip_num_cpu_threads,
        "log_to_console": model.log_to_console,
    }
    if model.absolute_dual_tolerance:
        solver_config["tolerances"][
            "absolute_dual"
        ] = model.absolute_dual_tolerance
    if model.relative_dual_tolerance:
        solver_config["tolerances"][
            "relative_dual"
        ] = model.relative_dual_tolerance
    if model.absolute_primal_tolerance:
        solver_config["tolerances"][
            "absolute_primal"
        ] = model.absolute_primal_tolerance
    if model.relative_primal_tolerance:
        solver_config["tolerances"][
            "relative_primal"
        ] = model.relative_primal_tolerance
    if model.absolute_gap_tolerance:
        solver_config["tolerances"][
            "absolute_gap"
        ] = model.absolute_gap_tolerance
    if model.relative_gap_tolerance:
        solver_config["tolerances"][
            "relative_gap"
        ] = model.relative_gap_tolerance
    if model.primal_infeasible_tolerance:
        solver_config["tolerances"][
            "primal_infeasible"
        ] = model.primal_infeasible_tolerance
    if model.dual_infeasible_tolerance:
        solver_config["tolerances"][
            "dual_infeasible"
        ] = model.dual_infeasible_tolerance
    if model.integrality_tolerance:
        solver_config["tolerances"][
            "integrality_tolerance"
        ] = model.integrality_tolerance
    if model.absolute_mip_gap:
        solver_config["tolerances"][
            "absolute_mip_gap"
        ] = model.absolute_mip_gap
    if model.relative_mip_gap:
        solver_config["tolerances"][
            "relative_mip_gap"
        ] = model.relative_mip_gap
    return solver_config


class SolverSettings:
    def __init__(self):
        self.absolute_dual_tolerance = None
        self.relative_dual_tolerance = None
        self.absolute_primal_tolerance = None
        self.relative_primal_tolerance = None
        self.absolute_gap_tolerance = None
        self.relative_gap_tolerance = None
        self.primal_infeasible_tolerance = None
        self.dual_infeasible_tolerance = None
        self.integrality_tolerance = None
        self.absolute_mip_gap = None
        self.relative_mip_gap = None
        self.detect_infeasibility = False
        self.time_limit = None
        self.iteration_limit = None
        self.solver_mode = SolverMode.Stable2
        self.pdlp_warm_start_data = None
        self.mip_incumbent_solution_callback = None
        self.mip_scaling = True
        self.mip_heuristics_only = None
        self.mip_num_cpu_threads = None
        self.log_to_console = True

    def set_optimality_tolerance(self, eps_optimal):
        """
        NOTE: Not supported for MILP, absolute is fixed to 1e-4,
              relative is fixed for 1e-6 and integrality is fixed for 1e-4.
              Dual is not supported for MILP.

        Set both absolute and relative tolerance on the primal feasibility,
        dual feasibility, and gap.
        Changing this value has a significant impact on accuracy and runtime.

        Optimality is computed as follows:

        dual_feasibility < absolute_dual_tolerance + relative_dual_tolerance
          * norm_objective_coefficient (l2_norm(c))
        primal_feasibility < absolute_primal_tolerance
          + relative_primal_tolerance * norm_constraint_bounds (l2_norm(b))
        duality_gap < absolute_gap_tolerance + relative_gap_tolerance
          * (abs(primal_objective) + abs(dual_objective))

        If all three conditions hold, optimality is reached.

        Parameters
        ----------
        eps_optimal : float64
            Tolerance to optimality

        Notes
        -----
        Default value is 1e-4.
        To set each absolute and relative tolerance, use the provided setters.
        """
        self.absolute_dual_tolerance = eps_optimal
        self.relative_dual_tolerance = eps_optimal
        self.absolute_primal_tolerance = eps_optimal
        self.relative_primal_tolerance = eps_optimal
        self.absolute_gap_tolerance = eps_optimal
        self.relative_gap_tolerance = eps_optimal

    def set_absolute_dual_tolerance(self, absolute_dual_tolerance):
        """
        NOTE: Not supported and not applicable for MILP.

        Set the absolute dual tolerance.

        Parameters
        ----------
        absolute_dual_tolerance : float64
            Absolute dual tolerance

        Notes
        -----
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.
        Default value is 1e-4.
        """
        self.absolute_dual_tolerance = absolute_dual_tolerance

    def set_relative_dual_tolerance(self, relative_dual_tolerance):
        """
        NOTE: Not supported and not applicable for MILP.

        Set the relative dual tolerance.

        Parameters
        ----------
        relative_dual_tolerance : float64
            Relative dual tolerance

        Notes
        -----
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.
        Default value is 1e-4.
        """
        self.relative_dual_tolerance = relative_dual_tolerance

    def set_absolute_primal_tolerance(self, absolute_primal_tolerance):
        """
        NOTE: Default values
            LP : 1e-4 and MILP : 1e-4.

        Set the absolute primal tolerance.

        Parameters
        ----------
        absolute_primal_tolerance : float64
            Absolute primal tolerance

        Notes
        -----
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.
        """
        self.absolute_primal_tolerance = absolute_primal_tolerance

    def set_relative_primal_tolerance(self, relative_primal_tolerance):
        """
        NOTE: Default values
            LP : 1e-4 and MILP : 1e-6.

        Set the relative primal tolerance.

        Parameters
        ----------
        relative_primal_tolerance : float64
            Relative primal tolerance

        Notes
        -----
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.
        """
        self.relative_primal_tolerance = relative_primal_tolerance

    def set_absolute_gap_tolerance(self, absolute_gap_tolerance):
        """
        NOTE: Not supported and not applicable for MILP.
        Default value is 1e-4.

        Set the absolute gap tolerance.

        Parameters
        ----------
        absolute_gap_tolerance : float64
            Absolute gap tolerance

        Notes
        -----
        For more details on tolerance to gap, see
        `set_optimality_tolerance` method.
        """
        self.absolute_gap_tolerance = absolute_gap_tolerance

    def set_relative_gap_tolerance(self, relative_gap_tolerance):
        """
        NOTE: Not supported and not applicable for MILP.
        Default value is 1e-4.

        Set the relative gap tolerance.

        Parameters
        ----------
        relative_gap_tolerance : float64
            Relative gap tolerance

        Notes
        -----
        For more details on tolerance to gap, see
        `set_optimality_tolerance` method.
        """
        self.relative_gap_tolerance = relative_gap_tolerance

    def set_infeasibility_detection(self, detect):
        """
        NOTE: Not supported for MILP.

        Solver will detect and leave if the problem is detected as infeasible.

        Parameters
        ----------
        detect : bool
            True to detect infeasibility, false to ignore it.

        Notes
        -----
        By default, the solver will not detect infeasibility.
        Some problems detected as infeasible may converge under a different
        tolerance factor.
        Detecting infeasibility consumes both runtime and memory. The added
        runtime is between 3% and 7%, added memory is between 10% and 20%.
        """
        self.detect_infeasibility = detect

    def set_primal_infeasible_tolerance(self, primal_infeasible_tolerance):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-8.

        Set the primal infeasible tolerance.

        Parameters
        ----------
        primal_infeasible_tolerance : float64
            Primal infeasible tolerance.

        Notes
        -----
        Higher values will detect infeasibility quicker but may trigger false
        positive.
        """
        self.primal_infeasible_tolerance = primal_infeasible_tolerance

    def set_dual_infeasible_tolerance(self, dual_infeasible_tolerance):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-8.

        Set the dual infeasible tolerance.

        Parameters
        ----------
        dual_infeasible_tolerance : float64
            Dual infeasible tolerance.

        Notes
        -----
        Higher values will detect infeasibility quicker but may trigger false
        positive.
        """
        self.dual_infeasible_tolerance = dual_infeasible_tolerance

    def set_integrality_tolerance(self, integrality_tolerance):
        """
        NOTE: Supported for MILP only
        Default value is 1e-5.

        Set integrality tolerance.

        Parameters
        ----------
        integrality_tolerance : float64
            Integrality tolerance

        Notes
        -----
        Default value is 1e-5.
        """
        self.integrality_tolerance = integrality_tolerance

    def set_absolute_mip_gap(self, absolute_mip_gap):
        """
        Set the MIP gap absolute tolerance.
        NOTE: Supported for MILP only

        Parameters
        ----------
        absolute_mip_gap : float64
            MIP gap absolute tolerance

        Notes
        -----
        Default value is 0.

        Solver will stop when the absolute gap is less than or
        equal to the specified absolute_mip_gap.

        The absolute gap is computed as follows:
        absolute gap = abs(best bound - best integer feasible solution)
        """
        self.absolute_mip_gap = absolute_mip_gap

    def set_relative_mip_gap(self, relative_mip_gap):
        """
        Set the MIP gap relative tolerance.
        NOTE: Supported for MILP only

        Parameters
        ----------
        relative_mip_gap : float64
            MIP gap relative tolerance

        Notes
        -----
        Default value is 1e-3.

        Solver will stop when the relative gap is less than or
        equal to the the specified relative_mip_gap.

        The relative gap is computed as follows:
        relative gap = abs(best bound - best integer feasible solution) /
                        abs(best integer feasible solution)
        """
        self.relative_mip_gap = relative_mip_gap

    def set_iteration_limit(self, iteration_limit):
        """
        NOTE: Not supported for MILP.

        Set the iteration limit after which the solver will stop and return the
        current solution.

        Parameters
        ----------
        iteration_limit : int
            Iteration limit to set.

        Notes
        -----
        By default there is no iteration limit.
        For performance reasons, cuOpt's does not constantly checks for
        iteration limit, thus, the solver might run a few extra iterations over
        the limit.
        If set along time limit, the first limit reached will exit.


        """
        self.iteration_limit = iteration_limit

    def set_time_limit(self, time_limit):
        """
        Set the time limit in seconds after which the solver will stop and
        return the current solution.
        If set along iteration limit, the first limit reached will exit.


        LP: Solver runs until optimality is reached within the time limit.
        If it does, it will return and will not wait for the entire
        duration of the time limit.

        MILP: Solver runs the entire duration of the time limit to search
        for a better solution.

        Parameters
        ----------
        time_limit : float64
            Time limit to set in seconds.

        Notes
        -----
        By default there is no time limit.
        For performance reasons, cuOpt's does not constantly checks for time
        limit, thus, the solver might run a few milliseconds over the limit.

        """
        self.time_limit = time_limit

    def set_pdlp_solver_mode(self, solver_mode):
        """
        NOTE: Not supported for MILP.

        Set the mode under which PDLP should operate.
        The mode will change the way PDLP internally optimizes the
        problem.
        The mode choice can drastically impact how fast a specific problem will
        be solved.
        Users are encouraged to test different modes to see which one fits the
        best their problem.
        By default, the solver uses SolverMode.Stable2, the best overall mode
        from our experiments.
        For now, only three modes are available : [Stable2, Methodical1, Fast1]

        Parameters
        ----------
        solver_mode : SolverMode
            Solver mode to set. Only possible values are:
            - SolverMode.Stable2
            - SolverMode.Methodical1
            - SolverMode.Fast1

        Notes
        -----
        For now, we don't offer any mechanism to know upfront which solver mode
        will be the best for one specific problem.
        Mode description:
        Stable2: Best compromise between success at converging and speed. If
        you want to use the legacy version, use Stable1.
        Methodical1: Usually leads to slower individual steps but less are
        needed to converge. It uses from 1.3x up to 1.7x times more memory
        Fast1: Less convergence success but usually yields the highest speed.
        """
        self.solver_mode = solver_mode

    def set_pdlp_warm_start_data(self, pdlp_warm_start_data):
        """
        Set the pdlp warm start data. This allows to restart PDLP with a
        previous solution context.

        This should be used when you solve a new problem which is similar to
        the previous one.

        Parameters
        ----------
        pdlp_warm_start_data : PDLPWarmStartData
            PDLP warm start data.

        Notes
        -----
        For now, the problem must have the same number of variables and
        constraints as the one found in the previous solution.

        Only supported solver modes are Stable2 and Fast1.

        Examples
        --------
        >>> solution = solver.Solve(first_problem, settings)
        >>> settings.set_pdlp_warm_start_data(
        >>>     solution.get_pdlp_warm_start_data()
        >>> )
        >>> solution = solver.Solve(second_problem, settings)
        """
        self.pdlp_warm_start_data = pdlp_warm_start_data

    def set_mip_incumbent_solution_callback(self, callback):
        """
        Note: Only supported for MILP

        Set the callback to receive incumbent solution.

        Parameters
        ----------
        callback : class for function callback

            Example is as shown below,

        Examples
        --------
        >>> class CustomLPIncumbentSolCallback(LPIncumbentSolCallback):
        >>>
        >>>     def __init__(self, sender, req_id):
        >>>        super().__init__()
        >>>        self.solution = None
        >>>        self.solution_cost = None
        >>>
        >>>
        >>>     def set_solution(self, solution, solution_cost):
        >>>         # This is numba array
        >>>         self.solution = solution.copy_to_host()
        >>>         self.solution_cost = solution_cost
        """
        self.mip_incumbent_solution_callback = callback

    def set_mip_scaling(self, mip_scaling):
        """
        Note: Only supported for MILP

        Get whether or not MIP problem scaling is enabled.

        Paramters
        ----------
        mip_scaling : bool
            True to enable MIP scaling, False to disable.

        Notes
        -----
        The feasibility may not match between the scaled and unscaled problem
        on numerically challenging models.
        Default value is True.
        """
        self.mip_scaling = mip_scaling

    def set_mip_heuristics_only(self, heuristics_only):
        """
        Set the heuristics only flag.

        Parameters
        ----------
        heuristics_only : bool
            True to run heuristics only,
            False to run heuristics and branch and bound.

        Notes
        -----
        By default, the solver runs both heuristics and branch and bound.
        """
        self.mip_heuristics_only = heuristics_only

    def get_mip_heuristics_only(self):
        """
        Get the heuristics only flag.
        """
        return self.mip_heuristics_only

    def set_mip_num_cpu_threads(self, num_cpu_threads):
        """
        Set the number of CPU threads to use for the branch and bound.

        Parameters
        ----------
        num_cpu_threads : int
            Number of CPU threads to use.
        """
        self.mip_num_cpu_threads = num_cpu_threads

    def get_mip_num_cpu_threads(self):
        """
        Get the number of CPU threads to use for the branch and bound.
        """
        return self.mip_num_cpu_threads

    def set_log_to_console(self, log_to_console):
        """
        Set the log to console flag.
        """
        self.log_to_console = log_to_console

    def get_log_to_console(self):
        """
        Get the log to console flag.
        """
        return self.log_to_console

    def get_mip_incumbent_solution_callback(self):
        """
        Return callback class object
        """
        return self.mip_incumbent_solution_callback

    def get_mip_scaling(self):
        """
        Note: Only supported for MILP

        Get whether or not MIP problem scaling is enabled.

        Paramters
        ----------
        enable_scaling : bool
            True to enable MIP scaling, False to disable.

        Notes
        -----
        The feasibility may not match between the scaled and unscaled problem
        on numerically challenging models.
        Default value is True.
        """
        return self.mip_scaling

    def get_absolute_dual_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-4.

        Get the absolute dual tolerance.
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The absolute dual tolerance.
        """
        return self.absolute_dual_tolerance

    def get_relative_dual_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-4.

        Get the relative dual tolerance.
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The relative dual tolerance.
        """
        return self.relative_dual_tolerance

    def get_absolute_primal_tolerance(self):
        """
        NOTE: Default values
            LP : 1e-4 and MILP : 1e-6.

        Get the absolute primal tolerance.
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The absolute primal tolerance.
        """
        return self.absolute_primal_tolerance

    def get_relative_primal_tolerance(self):
        """
        NOTE: Default values
            LP : 1e-4 and MILP : 1e-6.

        Get the relative primal tolerance.
        For more details on tolerance to optimality, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The relative primal tolerance.
        """
        return self.relative_primal_tolerance

    def get_absolute_gap_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-4.

        Get the absolute gap tolerance.
        For more details on tolerance to gap, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The absolute gap tolerance.
        """
        return self.absolute_gap_tolerance

    def get_relative_gap_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-4.

        Get the relative gap tolerance.
        For more details on tolerance to gap, see
        `set_optimality_tolerance` method.

        Returns
        -------
        float64
            The relative gap tolerance.
        """
        return self.relative_gap_tolerance

    def get_infeasibility_detection(self):
        """
        NOTE: Not supported for MILP.

        Get the status of detecting infeasibility.

        Returns
        -------
        bool
            Status of detecting infeasibility.
        """
        return self.detect_infeasibility

    def get_primal_infeasible_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-8.

        Get the primal infeasible tolerance.

        Returns
        -------
        float64
            The primal infeasible tolerance.
        """
        return self.primal_infeasible_tolerance

    def get_dual_infeasible_tolerance(self):
        """
        NOTE: Not supported for MILP.
        Default value is 1e-8.

        Get the dual infeasible tolerance.

        Returns
        -------
        float64
            The dual infeasible tolerance.
        """
        return self.dual_infeasible_tolerance

    def get_integrality_tolerance(self):
        """
        NOTE: Supported for MILP only
        Default value is 1e-5.

        Get integrality tolerance.
        """
        return self.integrality_tolerance

    def get_absolute_mip_gap(self):
        """
        NOTE: Supported for MILP only
        Default value is 0.

        Get the MIP gap absolute tolerance.
        """
        return self.absolute_mip_gap

    def get_relative_mip_gap(self):
        """
        NOTE: Supported for MILP only
        Default value is 1e-3.

        Get the MIP gap relative tolerance.
        """
        return self.relative_mip_gap

    def get_iteration_limit(self):
        """
        NOTE: Not supported for MILP.

        Get the iteration limit or None if none was set.

        Returns
        -------
        int or None
            The iteration limit.

        """
        return self.iteration_limit

    def get_time_limit(self):
        """
        Get the time limit in seconds or None if none was set.

        Returns
        -------
        float or None
            The time limit.
        """
        return self.time_limit

    def get_pdlp_solver_mode(self):
        """
        NOTE: Not supported for MILP.

        Get the solver mode.
        For more details on solver mode, see `set_pdlp_solver_mode` method.

        Returns
        -------
        Int
            The solver mode.
        """
        return self.solver_mode

    def get_pdlp_warm_start_data(self):
        """
        Returns the warm start data. See `set_pdlp_warm_start_data` for more
        details.

        Returns
        -------
        pdlp_warm_start_data

        Notes
        -----
        ...

        """
        return self.pdlp_warm_start_data
