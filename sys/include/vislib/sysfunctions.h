/*
 * sysfunctions.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSFUNCTIONS_H_INCLUDED
#define VISLIB_SYSFUNCTIONS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Answer the current working directory.
     *
     * @return The current working directory.
     *
     * @throws SystemException If the directory cannot be retrieved
     * @throws std::bad_alloc If there is not enough memory for storing the
     *                        directory.
     */
    StringA GetWorkingDirectoryA(void);

    /**
     * Answer the current working directory.
     *
     * @return The current working directory.
     *
     * @throws SystemException If the directory cannot be retrieved
     * @throws std::bad_alloc If there is not enough memory for storing the
     *                        directory.
     */
    StringW GetWorkingDirectoryW(void);

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */

