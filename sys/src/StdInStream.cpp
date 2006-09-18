/*
 * StdInStream.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/StdInStream.h"

#include <stdio.h>


/*
 * This file is intentionally empty.
 */
EXTENT vislib::sys::StdInStream::Read(void *buffer, EXTENT size) {
    return fread(buffer, 1, static_cast<size_t>(size), stdin);
}
