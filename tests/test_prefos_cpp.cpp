/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PreFOS/PreFOS.h>

#include <cstring>

int main()
{
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = nullptr;

    std::memset(&problem, 0, sizeof(problem));
    if (prefos_create_presolver(&problem, nullptr, &presolver) != PREFOS_STATUS_OK)
        return 1;
    if (prefos_run_presolve(presolver) != PREFOS_STATUS_OK)
    {
        prefos_free_presolver(presolver);
        return 1;
    }
    prefos_free_presolver(presolver);
    return 0;
}
