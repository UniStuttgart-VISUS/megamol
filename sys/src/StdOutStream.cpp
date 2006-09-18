/*
 * StdOutStream.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/StdOutStream.h"

#include <stdio.h>


/*
 * This file is intentionally empty.
 */
EXTENT vislib::sys::StdOutStream::Write(void *buffer, EXTENT size) {
    return fwrite(buffer, 1, static_cast<size_t>(size), stdout);
}
