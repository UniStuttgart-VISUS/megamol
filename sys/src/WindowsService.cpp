/*
 * WindowsService.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/WindowsService.h"

#ifdef _WIN32

// Inline file is required because current versions of VC++ and gcc seem to 
// generate code for templates with empty parameter list. Therefore, multiple
// definitions exist for every file that includes such a template, if the
// implementation is included in the header as usual.
// We use the inline file instead of placing the implementation in the cpp file
// directly as we cannot be sure that future compilers will behave the same.
#include "vislib/WindowsService.inl"

#endif /* _WIN32 */
