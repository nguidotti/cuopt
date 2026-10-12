# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Applies each patch to the git checkout SOURCE_DIR, skipping those already applied, so the
# FetchContent patch step also succeeds when it re-runs on an existing source tree.
#
#   cmake -DSOURCE_DIR=<dir> -P apply_patches.cmake <patch>...

if (NOT SOURCE_DIR)
    message(FATAL_ERROR "apply_patches.cmake: SOURCE_DIR is not set")
endif ()

math(EXPR last_arg "${CMAKE_ARGC} - 1")
set(first_patch "")
foreach (i RANGE ${last_arg})
    if ("${CMAKE_ARGV${i}}" STREQUAL "-P")
        math(EXPR first_patch "${i} + 2")
        break()
    endif ()
endforeach ()
if (first_patch STREQUAL "" OR first_patch GREATER last_arg)
    message(FATAL_ERROR "apply_patches.cmake: no patches given")
endif ()

foreach (i RANGE ${first_patch} ${last_arg})
    set(patch "${CMAKE_ARGV${i}}")
    execute_process(COMMAND git -C "${SOURCE_DIR}" apply --reverse --check "${patch}"
                    RESULT_VARIABLE not_applied OUTPUT_QUIET ERROR_QUIET)
    if (NOT not_applied)
        message(STATUS "Patch already applied: ${patch}")
        continue()
    endif ()
    execute_process(COMMAND git -C "${SOURCE_DIR}" apply "${patch}" RESULT_VARIABLE failed)
    if (failed)
        message(FATAL_ERROR "Failed to apply ${patch} to ${SOURCE_DIR}")
    endif ()
    message(STATUS "Applied patch: ${patch}")
endforeach ()
