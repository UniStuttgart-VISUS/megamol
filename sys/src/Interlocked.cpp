/*
 * Interlocked.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Interlocked.h"

#include "the/assert.h"
#include "the/not_supported_exception.h"



/*
 * vislib::sys::Interlocked::~Interlocked
 */
vislib::sys::Interlocked::~Interlocked(void) {
    // Should never be called.
    THE_ASSERT(false);
}


/*
 * vislib::sys::Interlocked::Interlocked
 */
vislib::sys::Interlocked::Interlocked(void) {
    throw the::not_supported_exception("Interlocked", __FILE__, __LINE__);
}
