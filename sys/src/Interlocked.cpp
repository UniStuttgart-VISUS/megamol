/*
 * Interlocked.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Interlocked.h"

#include "vislib/assert.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::sys::Interlocked::~Interlocked
 */
vislib::sys::Interlocked::~Interlocked(void) {
    // Should never be called.
    ASSERT(false);
}


/*
 * vislib::sys::Interlocked::Interlocked
 */
vislib::sys::Interlocked::Interlocked(void) {
    throw UnsupportedOperationException("Interlocked", __FILE__, __LINE__);
}
