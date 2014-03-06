/*
 * error.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ERROR_H_INCLUDED
#define VISLIB_ERROR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>


#else  /* _WIN32 */
#include <errno.h>
#include "the/system/system_error.h"
#include "the/types.h"

/**
 * Answer the last system error (::errno). This function is for enabling 
 * compatible error code handling under Windows and Linux.
 *
 * @return The last error.
 */
static inline the::system::system_error::native_error_type GetLastError(void) {
    return static_cast<the::system::system_error::native_error_type>(errno);
}

#endif /* !_WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ERROR_H_INCLUDED */
