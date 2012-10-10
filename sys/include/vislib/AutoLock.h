/*
 * AutoLock.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_AUTOLOCK_H_INCLUDED
#define VISLIB_AUTOLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ScopedLock.h"
#include "vislib/SyncObject.h"


namespace vislib {
namespace sys {

    /**
     * AutoLock is intended for backward compatibility.
     */
    typedef ScopedLock<SyncObject> AutoLock;
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_AUTOLOCK_H_INCLUDED */
