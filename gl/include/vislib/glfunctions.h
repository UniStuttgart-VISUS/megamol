/*
 * glfunctions.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLFUNCTIONS_H_INCLUDED
#define VISLIB_GLFUNCTIONS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/VersionNumber.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Answer the open gl version number. Note that the version number uses 
     * only the first two or the first three number elements (where the third
     * element 'buildNumber' is used as release number).
     *
     * @return The open gl version number.
     */
    const VersionNumber& GLVersion(void);

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLFUNCTIONS_H_INCLUDED */
