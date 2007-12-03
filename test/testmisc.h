/*
 * testmisc.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_TESTMISC_H_INCLUDED
#define VISLIBTEST_TESTMISC_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

/*
 * Test functions for less automated tests of smaller vislib functions 
 */

void TestConsoleColours(void);

void TestColumnFormatter(void);

void TestNetworkInformation(void);

void TestTrace(void);

void TestExceptions(void);

void TestSystemMessage(void);

void TestPerformanceCounter(void);

void TestPathManipulations(void);

void TestSingleLinkedListSort(void);

#endif /* VISLIBTEST_TESTMISC_H_INCLUDED */
