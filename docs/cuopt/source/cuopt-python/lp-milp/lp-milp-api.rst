=========================
LP and MILP API Reference
=========================

.. autoclass:: cuopt.linear_programming.problem.VType
   :members:
   :member-order: bysource
   :undoc-members:
   :exclude-members: capitalize, casefold, center, count, encode, endswith, expandtabs, find, format, format_map, index, isalnum, isalpha, isascii, isdecimal, isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper, join, ljust, lower, lstrip, maketrans, partition, removeprefix, removesuffix, replace, rfind, rindex, rjust, rpartition, rsplit, rstrip, split, splitlines, startswith, strip, swapcase, title, translate, upper, zfill

.. autoclass:: cuopt.linear_programming.problem.CType
   :members:
   :member-order: bysource
   :undoc-members:
   :exclude-members: capitalize, casefold, center, count, encode, endswith, expandtabs, find, format, format_map, index, isalnum, isalpha, isascii, isdecimal, isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper, join, ljust, lower, lstrip, maketrans, partition, removeprefix, removesuffix, replace, rfind, rindex, rjust, rpartition, rsplit, rstrip, split, splitlines, startswith, strip, swapcase, title, translate, upper, zfill

.. autoclass:: cuopt.linear_programming.problem.sense
   :members:
   :member-order: bysource
   :exclude-members: __new__, __init__, _generate_next_value_, as_integer_ratio, bit_count, bit_length, conjugate, denominator, from_bytes, imag, is_integer, numerator, real, to_bytes
   :no-inherited-members:

.. autoclass:: cuopt.linear_programming.problem.Problem
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: reset_solved_values, post_solve, dict_to_object, NumNZs, NumVariables, NumConstraints, IsMIP

.. autoclass:: cuopt.linear_programming.problem.Variable
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members:

.. autoclass:: cuopt.linear_programming.problem.LinearExpression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: cuopt.linear_programming.problem.Constraint
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: compute_slack
