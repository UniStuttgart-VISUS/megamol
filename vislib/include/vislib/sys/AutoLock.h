/*
 * AutoLock.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/sys/ScopedLock.h"
#include "vislib/sys/SyncObject.h"


namespace vislib::sys {

/**
 * AutoLock is intended for backward compatibility.
 */
typedef ScopedLock<SyncObject> AutoLock;

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
