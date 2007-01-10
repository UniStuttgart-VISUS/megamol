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
#include "vislib/String.h"


bool AssertTrue(const char *desc, const bool cond);

bool AssertFalse(const char *desc, const bool cond);

template<class T> bool AssertEqual(const char *desc, 
                                   const T& lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, (lhs == rhs));
}

template<class T> bool AssertEqual(const char *desc, 
                                   const char *lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, vislib::StringA(lhs).Compare(vislib::StringA(rhs)));
}

template<class T> bool AssertEqual(const char *desc, 
                                   const wchar_t *lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, vislib::StringW(lhs).Compare(vislib::StringW(rhs)));
}

template<class T> bool AssertEqualCaseInsensitive(const char *desc, 
                                   const char *lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, vislib::StringA(lhs).CompareInsensitive(vislib::StringA(rhs)));
}

template<class T> bool AssertEqualCaseInsensitive(const char *desc, 
                                   const wchar_t *lhs, 
                                   const T& rhs) {
    return ::AssertTrue(desc, vislib::StringW(lhs).CompareInsensitive(vislib::StringW(rhs)));
}

template<class T> bool AssertNotEqual(const char *desc,
                                      const T& lhs,
                                      const T& rhs) {
    return ::AssertTrue(desc, (lhs != rhs));
}

template<class T> bool AssertNearlyEqual(const char *desc, 
                                         const T& lhs, 
                                         const T& rhs) {
    return ::AssertTrue(desc, vislib::math::IsEqual<T>(lhs, rhs));
}

template<class T> bool AssertNotNearlyEqual(const char *desc,
                                            const T& lhs,
                                            const T& rhs) {
    return ::AssertFalse(desc, vislib::math::IsEqual<T>(lhs, rhs));
}

void AssertOutput(const char *desc);

void AssertOutputSuccess(void);

void AssertOutputFail(void);

void OutputAssertTestSummary(void);

// this succeeds if exactly the specified exception is thrown.
// has no return value!
#define AssertException(desc, call, exception) AssertOutput(desc); try { call; AssertOutputFail(); } catch(exception e) { AssertOutputSuccess(); } catch(...) { AssertOutputFail(); }

// this succeeds if NO exception is thrown.
// has no return value!
#define AssertNoException(desc, call) AssertOutput(desc); try { call; AssertOutputSuccess(); } catch(...) { AssertOutputFail(); }

#endif /* VISLIBTEST_TESTHELPER_H_INCLUDED */
