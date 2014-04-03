/*
 * PpmBitmapCodec.cpp
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#include "vislib/PpmBitmapCodec.h"
#include <climits>
#include "the/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/SystemInformation.h"
#include "the/not_implemented_exception.h"
#include "the/string.h"
#include "the/text/char_utility.h"
#include "the/text/string_utility.h"
#include "the/text/string_builder.h"


/*
 * vislib::graphics::PpmBitmapCodec::PpmBitmapCodec
 */
vislib::graphics::PpmBitmapCodec::PpmBitmapCodec(void)
        : AbstractBitmapCodec(), saveBinary(true) {
    // Intentionally empty
}


/*
 * vislib::graphics::PpmBitmapCodec::~PpmBitmapCodec
 */
vislib::graphics::PpmBitmapCodec::~PpmBitmapCodec(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::PpmBitmapCodec::AutoDetect
 */
int vislib::graphics::PpmBitmapCodec::AutoDetect(const void *mem,
        size_t size) const {
    if (size < 3) return -1; // insufficient preview data
    const char *data = static_cast<const char*>(mem);
    if ((data[0] != 'p') && (data[0] != 'P')) return 0; // wrong magic number
    if ((data[1] != '3') && (data[1] != '6') && (data[1] != 'F') && (data[1] != 'f')) return 0; // wrong magic number
    return the::text::char_utility::is_space(data[2])
        ? 1 // correctly terminated magic number
        : 0; // magic number not terminated
}


/*
 * vislib::graphics::PpmBitmapCodec::CanAutoDetect
 */
bool vislib::graphics::PpmBitmapCodec::CanAutoDetect(void) const {
    return true;
}


/*
 * vislib::graphics::PpmBitmapCodec::FileNameExtsA
 */
const char* vislib::graphics::PpmBitmapCodec::FileNameExtsA(void) const {
    return ".ppm;.pfm";
}


/*
 * vislib::graphics::PpmBitmapCodec::FileNameExtsW
 */
const wchar_t* vislib::graphics::PpmBitmapCodec::FileNameExtsW(void) const {
    return L".ppm;.pfm";
}


/*
 * vislib::graphics::PpmBitmapCodec::NameA
 */
const char * vislib::graphics::PpmBitmapCodec::NameA(void) const {
    return "Portable Pixmap / Floatmap";
}


/*
 * vislib::graphics::PpmBitmapCodec::NameW
 */
const wchar_t * vislib::graphics::PpmBitmapCodec::NameW(void) const {
    return L"Portable Pixmap / Floatmap";
}


/*
 * vislib::graphics::PpmBitmapCodec::loadFromMemory
 */
bool vislib::graphics::PpmBitmapCodec::loadFromMemory(const void *mem, size_t size) {
    BitmapImage& img = this->image();
    const char *cd = static_cast<const char*>(mem);
    size_t p1, p2;
    unsigned int w, h, v;
    the::astring tmp;
    float f, fac;

#define _LOCAL_PPM_SIFT(variable, type, method) \
    for (p1 = p2; (p1 < size) && the::text::char_utility::is_space(cd[p1]); p1++);\
    for (p2 = p1; (p2 < size) && !the::text::char_utility::is_space(cd[p2]); p2++);\
    if (p1 >= size) return false; /* out of data */ \
    variable = static_cast<type>(the::text::string_utility:: ## method(\
        the::astring(&cd[p1],\
        static_cast<size_t>(p2 - p1)).c_str()));

    if ((cd[0] != 'p') && (cd[0] != 'P')) return false; // wrong magic number
    if ((cd[1] != '3') && (cd[1] != '6') && (cd[1] != 'F') && (cd[1] != 'f')) return false; // wrong magic number
    if (!the::text::char_utility::is_space(cd[2])) return false;
        // magic number not terminated properly

    try {

        p2 = 2;
        _LOCAL_PPM_SIFT(w, unsigned int, parse_int)
        _LOCAL_PPM_SIFT(h, unsigned int, parse_int)
        _LOCAL_PPM_SIFT(v, unsigned int, parse_int)
        if (w < 0) return false; // width must be positive
        if (h < 0) return false; // width must be positive
        if (v <= 0) return false; // width must greater than zero

        if (cd[1] == '6') {
            // binary data

            if (v > 255) return false; // max value to large for binary files
            p2++;
            if (p2 + w * h * 3 > size) return false; // out of data

            img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_BYTE,
                static_cast<const void*>(&cd[p2]));
            img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
            img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
            img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);

            if (v < 255) {
                float f = 255.0f / static_cast<float>(v);
                for (unsigned int i = 0; i < 3 * w * h; i++) {
                    img.PeekDataAs<uint8_t>()[i] = static_cast<uint8_t>(
                        static_cast<float>(img.PeekDataAs<uint8_t>()[i]) * f);
                }
            }

            return true;

        } else if (cd[1] == 'F' || cd[1] == 'f') {
            // float

            p2 = 2;
            double endian;
            _LOCAL_PPM_SIFT(w, unsigned int, parse_int)
            _LOCAL_PPM_SIFT(h, unsigned int, parse_int)
            _LOCAL_PPM_SIFT(endian, double, parse_double)

            p2++;

            if (cd[1] == 'F') {
                if (p2 + w * h * 3 * sizeof(float) > size) return false; // out of data

                img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_FLOAT, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
            } else {
                if (p2 + w * h * sizeof(float) > size) return false; // out of data

                img.CreateImage(w, h, 1, BitmapImage::CHANNELTYPE_FLOAT, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_GRAY);
            }
            float *fd = img.PeekDataAs<float>();
            const float *buf = reinterpret_cast<const float*>(&cd[p2]);

            sys::SystemInformation::Endianness machineEnd =
                sys::SystemInformation::SystemEndianness();
            // endian <= 0.0f means little endian.
            sys::SystemInformation::Endianness fileEnd = (endian <= 0.0f) 
                ? sys::SystemInformation::ENDIANNESS_LITTLE_ENDIAN
                : sys::SystemInformation::ENDIANNESS_BIG_ENDIAN;
            
            if (machineEnd != sys::SystemInformation::ENDIANNESS_LITTLE_ENDIAN
                && machineEnd != sys::SystemInformation::ENDIANNESS_BIG_ENDIAN) {
                    throw new the::not_implemented_exception(
                        "Your machine is too exotic for this implementation",
                        __FILE__, __LINE__);
            }
            if (fileEnd == machineEnd) {
                // endianness agrees with this machine
                memcpy(fd, buf, sizeof(float) * img.GetChannelCount() * w * h);
            } else {
                const uint8_t *srcBytes;
                uint8_t *destBytes;
                for (unsigned int i = 0; i < img.GetChannelCount() * w * h; i++) {
                    srcBytes = reinterpret_cast<const uint8_t*>(&buf[i]);
                    destBytes = reinterpret_cast<uint8_t*>(&fd[i]);
                    destBytes[0] = srcBytes[3];
                    destBytes[1] = srcBytes[2];
                    destBytes[2] = srcBytes[1];
                    destBytes[3] = srcBytes[0];
                }
            }

            return true;

        } else {
            // ascii data

            if (v <= 255) { // byte
                img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_BYTE, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
                uint8_t *bd = img.PeekDataAs<uint8_t>();

                if (v == 255) {
                    for (unsigned int i = 0; i < 3 * w * h; i++) {
                        _LOCAL_PPM_SIFT(bd[i], uint8_t, parse_int)
                    }
                } else {
                    fac = 255.0f / static_cast<float>(v);
                    for (unsigned int i = 0; i < 3 * w * h; i++) {
                        _LOCAL_PPM_SIFT(f, float, parse_double)
                        bd[i] = static_cast<uint8_t>(f * fac);
                    }
                }

                return true;

            } else if (v <= 65535) { // word
                img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_WORD, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
                uint16_t *wd = img.PeekDataAs<uint16_t>();

                fac = 65535.0f / static_cast<float>(v);
                for (unsigned int i = 0; i < 3 * w * h; i++) {
                    _LOCAL_PPM_SIFT(f, float, parse_double)
                    wd[i] = static_cast<uint16_t>(f * fac);
                }

                return true;

            } else { // float
                img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_FLOAT, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
                float *fd = img.PeekDataAs<float>();

                fac = 1.0f / static_cast<float>(v); // normalise the data
                for (unsigned int i = 0; i < 3 * w * h; i++) {
                    _LOCAL_PPM_SIFT(f, float, parse_double)
                    fd[i] = f * fac;
                }

                return true;

            }
        }

    } catch(...) {
    }

#undef _LOCAL_PPM_SIFT

    return false;
}


/*
 * vislib::graphics::PpmBitmapCodec::loadFromMemoryImplemented
 */
bool vislib::graphics::PpmBitmapCodec::loadFromMemoryImplemented(void) const {
    return true;
}


/*
 * vislib::graphics::PpmBitmapCodec::saveToMemory
 */
bool vislib::graphics::PpmBitmapCodec::saveToMemory(vislib::RawStorage& outmem) const {
    const BitmapImage& img = this->image();
    bool bin = this->saveBinary;
    unsigned int cc = img.GetChannelCount();
    unsigned int cr = UINT_MAX, cg = UINT_MAX, cb = UINT_MAX;
    unsigned int maxVal = 255;
    const uint8_t *bd = img.PeekDataAs<uint8_t>();
    const uint16_t *wd = NULL;
    size_t headLen;
    unsigned int imgSize = img.Width() * img.Height();

    for (unsigned int i = 0; i < cc; i++) {
        switch (img.GetChannelLabel(i)) {
            case BitmapImage::CHANNEL_RED: cr = i; break;
            case BitmapImage::CHANNEL_GREEN: cg = i; break;
            case BitmapImage::CHANNEL_BLUE: cb = i; break;
            case BitmapImage::CHANNEL_GRAY:
                if ((cr == UINT_MAX) && (cg == UINT_MAX) && (cb == UINT_MAX)) {
                    cr = cg = cb = i;
                }
                break;
            case BitmapImage::CHANNEL_ALPHA:
                if ((cr == UINT_MAX) && (cg == UINT_MAX) && (cb == UINT_MAX)) {
                    cr = cg = cb = i;
                }
                break;
#ifndef _WIN32
            default: break;
#endif /* !_WIN32 */
        }
    }

    if (img.GetChannelType() == BitmapImage::CHANNELTYPE_WORD) {
        maxVal = 65535;
        bin = false;
        bd = NULL;
        wd = img.PeekDataAs<uint16_t>();

    } else if (img.GetChannelType() == BitmapImage::CHANNELTYPE_FLOAT) {
        // write a PFM

        sys::SystemInformation::Endianness machineEnd =
            sys::SystemInformation::SystemEndianness();

        if (img.GetChannelCount() != 1 && img.GetChannelCount() != 3) {
            throw new the::not_implemented_exception(
                "This implementation can only cope with RGB and greyscale images",
                __FILE__, __LINE__);
        }
        if (machineEnd != sys::SystemInformation::ENDIANNESS_BIG_ENDIAN
            && machineEnd != sys::SystemInformation::ENDIANNESS_LITTLE_ENDIAN) {
                throw new the::not_implemented_exception(
                    "Your machine is too exotic for this implementation",
                    __FILE__, __LINE__);
        }

        the::astring data;
        the::astring tmp;
        the::text::astring_builder::format_to(data, "P%c\n%u %u\n%f\n", img.GetChannelCount() == 1 ? 'f' : 'F', img.Width(), img.Height(), 
            machineEnd == sys::SystemInformation::ENDIANNESS_BIG_ENDIAN ? 1.0f : -1.0f);

        size_t imgLen = img.GetChannelCount() * img.Width() * img.Height() * sizeof(float);
        size_t bodyLen = data.size() + imgLen;
        outmem.EnforceSize(bodyLen);
        memcpy(outmem, data.c_str(), data.size());
        memcpy(outmem.At(data.size()), img.PeekDataAs<uint8_t>(), imgLen);

        return true;
    }

    the::astring header;
    the::text::astring_builder::format_to(header, "P%d\n%u %u\n%u\n",
        bin ? 6 : 3, img.Width(), img.Height(), maxVal);
    outmem.EnforceSize(headLen = header.size());
    memcpy(outmem, header.c_str(), headLen);

    if (bin) {
        THE_ASSERT(bd != NULL);
        outmem.AssertSize(headLen + imgSize * 3, true);
        for (unsigned int i = 0; i < imgSize; i++) {
            outmem.AsAt<uint8_t>(headLen)[i * 3]
                = (cr != UINT_MAX) ? bd[i * cc + cr] : 0;
            outmem.AsAt<uint8_t>(headLen)[i * 3 + 1]
                = (cg != UINT_MAX) ? bd[i * cc + cg] : 0;
            outmem.AsAt<uint8_t>(headLen)[i * 3 + 2]
                = (cb != UINT_MAX) ? bd[i * cc + cb] : 0;
        }

    } else {
        the::astring data;
        the::astring tmp;
        unsigned int ppl; // pixel per line

        the::text::astring_builder::format_to(tmp, "%u %u %u ", maxVal, maxVal, maxVal);
        ppl = static_cast<unsigned int>(70 / tmp.size());
        if (ppl < 1) ppl = 1;

        if (bd != NULL) {
            for (unsigned int i = 0; i < imgSize; i++) {
                the::text::astring_builder::format_to(tmp, "%u %u %u%c",
                    (cr != UINT_MAX) ? bd[i * cc + cr] : 0,
                    (cg != UINT_MAX) ? bd[i * cc + cg] : 0,
                    (cb != UINT_MAX) ? bd[i * cc + cb] : 0,
                    (((i + 1) % ppl) == 0) ? '\n' : ' ');
                data += tmp; // slow!
            }
        } else if (wd != NULL) {
            for (unsigned int i = 0; i < imgSize; i++) {
                the::text::astring_builder::format_to(tmp, "%u %u %u%c",
                    (cr != UINT_MAX) ? wd[i * cc + cr] : 0,
                    (cg != UINT_MAX) ? wd[i * cc + cg] : 0,
                    (cb != UINT_MAX) ? wd[i * cc + cb] : 0,
                    (((i + 1) % ppl) == 0) ? '\n' : ' ');
                data += tmp; // slow!
            }
        } else return false; // internal format error

        size_t bodyLen = data.size();
        outmem.AssertSize(headLen + bodyLen, true);
        memcpy(outmem.At(headLen), data.c_str(), bodyLen);
    }

    return true;
}


/*
 * vislib::graphics::PpmBitmapCodec::saveToMemoryImplemented
 */
bool vislib::graphics::PpmBitmapCodec::saveToMemoryImplemented(void) const {
    return true;
}
