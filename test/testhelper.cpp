/*
 * testhelper.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include <iostream>

#include "testhelper.h"


bool AssertTrue(const char *desc, const bool cond) {
    if (cond) {
        std::cout << "\"" << desc << "\" succeeded." << std::endl;
    } else {
        std::cout << "\n\"" << desc << "\" FAILED.\n" << std::endl;
    }

    return cond;
}


bool AssertFalse(const char *desc, const bool cond) {
    return ::AssertTrue(desc, !cond);
}