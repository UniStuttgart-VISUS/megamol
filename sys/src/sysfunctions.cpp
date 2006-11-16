/*
 * sysfunctions.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/sysfunctions.h"

#ifdef _WIN32
#else /* _WIN32 */
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/error.h"
#include "vislib/memutils.h"
#include "vislib/SystemException.h"


/*
 * vislib::sys::GetWorkingDirectoryA
 */
vislib::StringA vislib::sys::GetWorkingDirectoryA(void) {
#ifdef _WIN32
#else /* _WIN32 */
    const SIZE_T BUFFER_GROW = 32;
    SIZE_T bufferSize = 256;
    char *buffer = new char[bufferSize];

    while (::getcwd(buffer, bufferSize) == NULL) {
        if (errno == ERANGE) {
            ARY_SAFE_DELETE(buffer);
            bufferSize += BUFFER_GROW;
            buffer = new char[bufferSize];
        } else {
            throw SystemException(errno, __FILE__, __LINE__);
        }
    }

    StringA retval(buffer);
    ARY_SAFE_DELETE(buffer);
    return retval;

#endif /* _WIN32 */
}


/*
 * vislib::sys::GetWorkingDirectoryW
 */
vislib::StringW vislib::sys::GetWorkingDirectoryW(void) {
#ifdef _WIN32
#else /* _WIN32 */
    return StringW(GetWorkingDirectoryA());
#endif /* _WIN32 */
}