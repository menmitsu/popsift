/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#define POPSIFT_IS_DEFINED(F) F() == 1

#define POPSIFT_HAVE_SHFL_DOWN_SYNC() 1
#define POPSIFT_HAVE_NORMF()          0
#define POPSIFT_DISABLE_GRID_FILTER() 0
#define POPSIFT_USE_NVTX()            0


