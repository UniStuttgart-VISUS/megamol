/*
 * utils.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/utils.h"


/*
 * vislib::UIntRLEEncode
 */
bool vislib::UIntRLEEncode(unsigned char *dst, unsigned int &len, UINT64 src) {
    unsigned int pos = 0;
    if (src == 0) {
        if (len == 0) return false; // dst insufficient
        dst[0] = 0;
        pos = 1;
    } else while (src > 0) {
        if (pos == len) return false; // dst insufficient
        dst[pos++] = (src & 0x7F) + ((src > 128) ? 128 : 0);
        src >>= 7;
    }
    len = pos;
    return true;
}


/*
 * vislib::UIntRLEDecode
 */
bool vislib::UIntRLEDecode(UINT64 &dst, unsigned char *src, unsigned int &len) {
    unsigned int pos = 0;
    UINT64 mult = 0;
    dst = 0;
    do {
        if ((pos == 10) // would be reading the 11th byte
            || (pos == len)) return false; // insufficient data
        dst += (static_cast<UINT64>(src[pos++] & 0x7F) << mult);
        mult += 7;
    } while (src[pos - 1] >= 128);
    len = pos;
    return true;
}
