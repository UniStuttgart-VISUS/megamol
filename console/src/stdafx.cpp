/*
 * stdafx.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/String.h"


/*
 * ASCIIStringTM
 */
const char* ASCIIStringTM(void) {
    static vislib::StringA tmA;
    if (tmA.IsEmpty()) {
        vislib::StringW tm(0x2122, 1);
#ifndef _WIN32
        tmA = tm;
        if (!vislib::StringW(tmA).Equals(tm)) {
#endif /* !_WIN32 */
            tmA = "(TM)";
#ifndef _WIN32
        }
#endif /* !_WIN32 */
    }
    return tmA.PeekBuffer();
}
