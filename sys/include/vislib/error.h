/*
 * error.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ERROR_H_INCLUDED
#define VISLIB_ERROR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifdef _WIN32
#include <windows.h>

#else  /* _WIN32 */
#include <errno.h>

#include "vislib/types.h"

/**
 * Answer the last system error (::errno). This function is for enabling 
 * compatible error code handling under Windows and Linux.
 *
 * @return The last error.
 */
static inline DWORD GetLastError(void) {
    return static_cast<DWORD>(errno);
}

#endif /* !_WIN32 */

#endif /* VISLIB_ERROR_H_INCLUDED */
