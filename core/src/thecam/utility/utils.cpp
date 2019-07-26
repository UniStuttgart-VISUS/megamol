/*
 * the/utils.cpp
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * utils.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "mmcore/thecam/utility/utils.h"
#include "mmcore/thecam/utility/assert.h"


/*
 * megamol::core::thecam::utility::uint_rle_encode
 */
bool megamol::core::thecam::utility::uint_rle_encode(unsigned char* dst, unsigned int& len, uint64_t src) {
    unsigned int pos = 0;
    if (src == 0) {
        if (len == 0) return false; // dst insufficient
        dst[0] = 0;
        pos = 1;
    } else
        while (src > 0) {
            if (pos == len) return false; // dst insufficient
            dst[pos++] = (src & 0x7F) + ((src > 128) ? 128 : 0);
            src >>= 7;
        }
    len = pos;
    return true;
}


/*
 * megamol::core::thecam::utility::uint_rle_decode
 */
bool megamol::core::thecam::utility::uint_rle_decode(uint64_t& dst, unsigned char* src, unsigned int& len) {
    unsigned int pos = 0;
    uint64_t mult = 0;
    dst = 0;
    do {
        if ((pos == 10) // would be reading the 11th byte
            || (pos == len))
            return false; // insufficient data
        dst += (static_cast<uint64_t>(src[pos++] & 0x7F) << mult);
        mult += 7;
    } while (src[pos - 1] >= 128);
    len = pos;
    return true;
}
