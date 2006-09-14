/*
 * testhelper.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_TESTHELPER_H_INCLUDED
#define VISLIBTEST_TESTHELPER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include <iostream>

#include "vislib/types.h"
#include "vislib/mathfunctions.h"


bool AssertTrue(const char *desc, const bool cond);

bool AssertFalse(const char *desc, const bool cond);

template<class T> bool AssertEqual(const char *desc, 
                                   const T& lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, (lhs == rhs));
}

template<class T> bool AssertNotEqual(const char *desc,
                                      const T& lhs,
                                      const T& rhs) {
    return ::AssertTrue(desc, (lhs != rhs));
}

template<class T> bool AssertNearlyEqual(const char *desc, 
                                         const T& lhs, 
                                         const T& rhs) {
    return ::AssertTrue(desc, vislib::math::FltEqual(lhs, rhs));
}

template<class T> bool AssertNotNearlyEqual(const char *desc,
                                            const T& lhs,
                                            const T& rhs) {
    return ::AssertFalse(desc, vislib::math::FltEqual(lhs, rhs));
}

#endif /* VISLIBTEST_TESTHELPER_H_INCLUDED */
