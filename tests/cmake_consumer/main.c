/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PreFOS/PreFOS.h>

#include <string.h>

int main(void)
{
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    if (PREFOS_VERSION_MAJOR != 0 || PREFOS_VERSION_MINOR != 1) return 1;
    if (prefos_create_presolver(&problem, NULL, &presolver) != PREFOS_STATUS_OK)
        return 1;
    prefos_free_presolver(presolver);
    return 0;
}
