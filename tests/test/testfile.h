/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_TESTFILE_H_INCLUDED
#define VISLIBTEST_TESTFILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

void TestFile(void);

void TestBaseFile(void);

void TestBufferedFile(void);

void TestMemmappedFile(void);

void TestPath(void);

#endif /* VISLIBTEST_TESTFILE_H_INCLUDED */
