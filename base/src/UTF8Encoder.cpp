/*
 * UTF8Encoder.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/UTF8Encoder.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/StringConverter.h"
#include "vislib/CharTraits.h"

#include <iostream>

#define __SUPPORT_SIX_OCTETS__ 1


/*
 * vislib::UTF8Encoder::UTF8Encoder
 */
vislib::UTF8Encoder::UTF8Encoder(void) {
    throw UnsupportedOperationException("vislib::UTF8Encoder::Ctor", __FILE__, __LINE__);
}


/*
 * vislib::UTF8Encoder::~UTF8Encoder
 */
vislib::UTF8Encoder::~UTF8Encoder(void) {
}


/*
 * vislib::UTF8Encoder::CalcUTF8Size
 */
vislib::UTF8Encoder::Size vislib::UTF8Encoder::CalcUTF8Size(const char *str) {
    Size size = 1; // terminating zero

    if (str != NULL) {
        const unsigned char *ucb = reinterpret_cast<const unsigned char*>(str);

        while (*ucb != 0) {
            if (*ucb < 128) {
                // std ANSI needs a single byte
                size += 1; 

            } else {
                // extended ANSI must be treated like Unicode-16
                wchar_t w = CharTraitsA::ToUnicode(*ucb);

                if (w < 128) { // 7 Bit : 2^7=128
                    size += 1; // should never be called
                } else if (w < 2048) { // 5 + 6 = 11 Bit : 2^11=2048
                    size += 2;
                } else if (w < 65536) { // 4 + 6 + 6 = 16 Bit : 2^16=65536
                    size += 3;
                } else if (w < 2097152) { // 3 + 6 + 6 + 6 = 21 Bit : 2^21=2097152
                    size += 4;
#ifdef __SUPPORT_SIX_OCTETS__
                } else if (w < 67108864) { // 2 + 6 + 6 + 6 + 6 = 26 Bit : 2^26=67108864
                    size += 5; // Not too unicode compatible!
                } else if (w < 2097152) { // 1 + 6 + 6 + 6 + 6 + 6 = 31 Bit : 2^31=2147483648
                    size += 6; // Not too unicode compatible!
#endif /* __SUPPORT_SIX_OCTETS__ */
                } else { 
                    return -size; // error, not representable character
                }

            }

            ucb++;
        }
    }

    return size;
}


/*
 * vislib::UTF8Encoder::CalcUTF8Size
 */
vislib::UTF8Encoder::Size vislib::UTF8Encoder::CalcUTF8Size(const wchar_t *str) {
    Size size = 1; // terminating zero

    if (str != NULL) {
        while (*str != 0) {

            if (*str < 128) { // 7 Bit : 2^7=128
                size += 1; 
            } else if (*str < 2048) { // 5 + 6 = 11 Bit : 2^11=2048
                size += 2;
            } else if (*str < 65536) { // 4 + 6 + 6 = 16 Bit : 2^16=65536
                size += 3;
            } else if (*str < 2097152) { // 3 + 6 + 6 + 6 = 21 Bit : 2^21=2097152
                size += 4;
#ifdef __SUPPORT_SIX_OCTETS__
            } else if (*str < 67108864) { // 2 + 6 + 6 + 6 + 6 = 26 Bit : 2^26=67108864
                size += 5; // Not too unicode compatible!
            } else if (*str < 2097152) { // 1 + 6 + 6 + 6 + 6 + 6 = 31 Bit : 2^31=2147483648
                size += 6; // Not too unicode compatible!
#endif /* __SUPPORT_SIX_OCTETS__ */
            } else { 
                return -size; // error, not representable character
            }

            str++;
        }
    }

    return size;
}


/*
 * vislib::UTF8Encoder::StringLength
 */
vislib::UTF8Encoder::Size vislib::UTF8Encoder::StringLength(const char *str) {
    Size size = 0;

    if (str != NULL) {
        const unsigned char *ucb = reinterpret_cast<const unsigned char*>(str);

        while (*ucb != 0) {
            if (*ucb > 127) {
                // multibyte character
                int ro; // remaining octets

                if ((*ucb & 0xE0) == 0xC0) { 
                    ro = 1; 
                } else if ((*ucb & 0xF0) == 0xE0) { 
                    ro = 2; 
                } else if ((*ucb & 0xF8) == 0xF0) { 
                    ro = 3; 
                } else if ((*ucb & 0xFC) == 0xF8) { 
                    ro = 4; 
                } else if ((*ucb & 0xFE) == 0xFC) { 
                    ro = 5; 
                } else {
                    return -1; // invalid starting bit pattern
                }

                for (;ro > 0; ro--) {
                    ucb++;
                    if ((*ucb & 0xC0) != 0x80) {
                        return -2; // invalid block bit pattern.
                    }
                }
            }

            size++;
            ucb++;
        }
    }

    return size;
}


/*
 * vislib::UTF8Encoder::CalcUTF8Size
 */
bool vislib::UTF8Encoder::Encode(vislib::StringA& outTarget, const char *source) {
    unsigned int size = CalcUTF8Size(source);
    if (size <= 0) return false; // not representable characters included

    unsigned char *cb = reinterpret_cast<unsigned char*>(outTarget.AllocateBuffer(size - 1));

    if (size > 1) {
        UINT32 val;

        while(*source != 0) {

            if (*reinterpret_cast<const unsigned char*>(source) < 128) { // 7 Bit : 2^7=128
                *(cb++) = static_cast<unsigned char>(*source);

            } else {
                val = CharTraitsA::ToUnicode(*source);
                if (val < 2048) { // 5 + 6 = 11 Bit : 2^11=2048
                    // 110xxxxx 10xxxxxx
                    cb[1] = 0x80 | (val & 0x3F);
                    cb[0] = 0xC0 | (val >> 6);
                    cb += 2;

                } else if (val < 65536) { // 4 + 6 + 6 = 16 Bit : 2^16=65536
                    // 1110xxxx 10xxxxxx 10xxxxxx
                    cb[2] = 0x80 | (val & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xE0 | (val >> 6);
                    cb += 3;

                } else if (val < 2097152) { // 3 + 6 + 6 + 6 = 21 Bit : 2^21=2097152
                    // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[3] = 0x80 | (val & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xF0 | (val >> 6);
                    cb += 4;

#ifdef __SUPPORT_SIX_OCTETS__

                } else if (val < 67108864) { // 2 + 6 + 6 + 6 + 6 = 26 Bit : 2^26=67108864
                    // 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[4] = 0x80 | (val & 0x3F);
                    cb[3] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xF8 | (val >> 6);
                    cb += 5;

                } else if (val < 2097152) { // 1 + 6 + 6 + 6 + 6 + 6 = 31 Bit : 2^31=2147483648
                    // 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[5] = 0x80 | (val & 0x3F);
                    cb[4] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[3] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xFC | (val >> 6);
                    cb += 6;

#endif /* __SUPPORT_SIX_OCTETS__ */

                } else { 
                    return false; // error, not representable character

                }
            }

            source++;
        }
    }

    return true;
}


/*
 * vislib::UTF8Encoder::CalcUTF8Size
 */
bool vislib::UTF8Encoder::Encode(vislib::StringA& outTarget, const wchar_t *source) {
    unsigned int size = CalcUTF8Size(source);
    if (size <= 0) return false; // not representable characters included

    unsigned char *cb = reinterpret_cast<unsigned char*>(outTarget.AllocateBuffer(size - 1));

    if (size > 1) {
        UINT32 val;

        while(*source != 0) {

            if (*source < 128) { // 7 Bit : 2^7=128
                *(cb++) = static_cast<unsigned char>(*source);

            } else {
                val = *source;
                if (val < 2048) { // 5 + 6 = 11 Bit : 2^11=2048
                    // 110xxxxx 10xxxxxx
                    cb[1] = 0x80 | (val & 0x3F);
                    cb[0] = 0xC0 | (val >> 6);
                    cb += 2;

                } else if (val < 65536) { // 4 + 6 + 6 = 16 Bit : 2^16=65536
                    // 1110xxxx 10xxxxxx 10xxxxxx
                    cb[2] = 0x80 | (val & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xE0 | (val >> 6);
                    cb += 3;

                } else if (val < 2097152) { // 3 + 6 + 6 + 6 = 21 Bit : 2^21=2097152
                    // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[3] = 0x80 | (val & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xF0 | (val >> 6);
                    cb += 4;

#ifdef __SUPPORT_SIX_OCTETS__

                } else if (val < 67108864) { // 2 + 6 + 6 + 6 + 6 = 26 Bit : 2^26=67108864
                    // 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[4] = 0x80 | (val & 0x3F);
                    cb[3] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xF8 | (val >> 6);
                    cb += 5;

                } else if (val < 2097152) { // 1 + 6 + 6 + 6 + 6 + 6 = 31 Bit : 2^31=2147483648
                    // 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
                    cb[5] = 0x80 | (val & 0x3F);
                    cb[4] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[3] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[2] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[1] = 0x80 | ((val >>= 6) & 0x3F);
                    cb[0] = 0xFC | (val >> 6);
                    cb += 6;

#endif /* __SUPPORT_SIX_OCTETS__ */

                } else { 
                    return false; // error, not representable character

                }
            }

            source++;
        }
    }

    return true;
}


/*
 * vislib::UTF8Encoder::Decode
 */
bool vislib::UTF8Encoder::Decode(vislib::StringA& outTarget, const char *source) {
    Size s = StringLength(source);
    if (s < 0) return false;

    char *buf = outTarget.AllocateBuffer(s);
    if (s > 0) {
        const unsigned char *ucb = reinterpret_cast<const unsigned char*>(source);
        UINT32 val = 0;
        int ro; // remaining octets

        while (*ucb != 0) {
            if (*ucb < 128) {
                // singlebyte character 
                *(buf++) = *reinterpret_cast<const char*>(ucb);

            } else {
                // multibyte character

                if ((*ucb & 0xE0) == 0xC0) { 
                    val = *ucb & 0x1F;
                    ro = 1; 
                } else if ((*ucb & 0xF0) == 0xE0) { 
                    val = *ucb & 0x0F;
                    ro = 2; 
                } else if ((*ucb & 0xF8) == 0xF0) { 
                    val = *ucb & 0x07;
                    ro = 3; 
                } else if ((*ucb & 0xFC) == 0xF8) { 
                    val = *ucb & 0x03;
                    ro = 4; 
                } else if ((*ucb & 0xFE) == 0xFC) { 
                    val = *ucb & 0x01;
                    ro = 5; 
                } else {
                    return false; // invalid starting bit pattern
                }

                for (;ro > 0; ro--) {
                    ucb++;
                    val = (val << 6) | (*ucb & 0x3F);
                    if ((*ucb & 0xC0) != 0x80) {
                        return false; // invalid block bit pattern.
                    }
                }

                *(buf++) = CharTraitsW::ToANSI(val);

            }
            ucb++;
        }
        *buf = 0;
    }

    return true;
}


/*
 * vislib::UTF8Encoder::Decode
 */
bool vislib::UTF8Encoder::Decode(vislib::StringW& outTarget, const char *source) {
    Size s = StringLength(source);
    if (s < 0) return false;

    wchar_t *buf = outTarget.AllocateBuffer(s);
    if (s > 0) {
        const unsigned char *ucb = reinterpret_cast<const unsigned char*>(source);
        UINT32 val = 0;
        int ro; // remaining octets

        while (*ucb != 0) {
            if (*ucb < 128) {
                // singlebyte character 
                *(buf++) = *ucb; // lowest 7 bit are identical on ANSI and Unicode

            } else {
                // multibyte character

                if ((*ucb & 0xE0) == 0xC0) { 
                    val = *ucb & 0x1F;
                    ro = 1; 
                } else if ((*ucb & 0xF0) == 0xE0) { 
                    val = *ucb & 0x0F;
                    ro = 2; 
                } else if ((*ucb & 0xF8) == 0xF0) { 
                    val = *ucb & 0x07;
                    ro = 3; 
                } else if ((*ucb & 0xFC) == 0xF8) { 
                    val = *ucb & 0x03;
                    ro = 4; 
                } else if ((*ucb & 0xFE) == 0xFC) { 
                    val = *ucb & 0x01;
                    ro = 5; 
                } else {
                    return false; // invalid starting bit pattern
                }

                for (;ro > 0; ro--) {
                    ucb++;
                    val = (val << 6) | (*ucb & 0x3F);
                    if ((*ucb & 0xC0) != 0x80) {
                        return false; // invalid block bit pattern.
                    }
                }
        
                *(buf++) = val;

            }
            ucb++;
        }
        *buf = 0;
    }

    return true;
}
