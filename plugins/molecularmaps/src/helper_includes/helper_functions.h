/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// These are helper functions for the SDK samples (string parsing, timers, image helpers, etc)
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

// includes, project
#include <assert.h>
#include <exception.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

// includes, timer, string parsing, image helpers
#include <helper_image.h>  // helper functions for image compare, dump, data comparisons
#include <helper_string.h> // helper functions for string parsing
#include <helper_timer.h>  // helper functions for timers

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif //  HELPER_FUNCTIONS_H
