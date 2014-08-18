/*
 * PpmBitmapCodec.cpp
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#include "vislib/PpmBitmapCodec.h"
#include <climits>
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/mathfunctions.h"
#include "vislib/SystemInformation.h"
#include "vislib/MissingImplementationException.h"


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
        SIZE_T size) const {
    if (size < 3) return -1; // insufficient preview data
    const char *data = static_cast<const char*>(mem);
    if ((data[0] != 'p') && (data[0] != 'P')) return 0; // wrong magic number
    if ((data[1] != '3') && (data[1] != '6') && (data[1] != 'F') && (data[1] != 'f')) return 0; // wrong magic number
    return vislib::CharTraitsA::IsSpace(data[2])
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
bool vislib::graphics::PpmBitmapCodec::loadFromMemory(const void *mem, SIZE_T size) {
    BitmapImage& img = this->image();
    const char *cd = static_cast<const char*>(mem);
    SIZE_T p1, p2;
    unsigned int w, h, v;
    vislib::StringA tmp;
    float f, fac;

#define _LOCAL_PPM_SIFT(variable, type, method) \
    for (p1 = p2; (p1 < size) && vislib::CharTraitsA::IsSpace(cd[p1]); p1++);\
    for (p2 = p1; (p2 < size) && !vislib::CharTraitsA::IsSpace(cd[p2]); p2++);\
    if (p1 >= size) return false; /* out of data */ \
    variable = static_cast<type>(vislib::CharTraitsA::method(\
        vislib::StringA(&cd[p1],\
        static_cast<vislib::CharTraitsA::Size>(p2 - p1)).PeekBuffer()));

    if ((cd[0] != 'p') && (cd[0] != 'P')) return false; // wrong magic number
    if ((cd[1] != '3') && (cd[1] != '6') && (cd[1] != 'F') && (cd[1] != 'f')) return false; // wrong magic number
    if (!vislib::CharTraitsA::IsSpace(cd[2])) return false;
        // magic number not terminated properly

    try {

        p2 = 2;
        _LOCAL_PPM_SIFT(w, unsigned int, ParseInt)
        _LOCAL_PPM_SIFT(h, unsigned int, ParseInt)
        _LOCAL_PPM_SIFT(v, unsigned int, ParseInt)
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
                    img.PeekDataAs<BYTE>()[i] = static_cast<BYTE>(
                        static_cast<float>(img.PeekDataAs<BYTE>()[i]) * f);
                }
            }

            return true;

        } else if (cd[1] == 'F' || cd[1] == 'f') {
            // float

            p2 = 2;
            double endian;
            _LOCAL_PPM_SIFT(w, unsigned int, ParseInt)
            _LOCAL_PPM_SIFT(h, unsigned int, ParseInt)
            _LOCAL_PPM_SIFT(endian, double, ParseDouble)

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
                    throw new vislib::MissingImplementationException(
                        "Your machine is too exotic for this implementation",
                        __FILE__, __LINE__);
            }
            if (fileEnd == machineEnd) {
                // endianness agrees with this machine
                memcpy(fd, buf, sizeof(float) * img.GetChannelCount() * w * h);
            } else {
                const UINT8 *srcBytes;
                UINT8 *destBytes;
                for (unsigned int i = 0; i < img.GetChannelCount() * w * h; i++) {
                    srcBytes = reinterpret_cast<const UINT8*>(&buf[i]);
                    destBytes = reinterpret_cast<UINT8*>(&fd[i]);
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
                BYTE *bd = img.PeekDataAs<BYTE>();

                if (v == 255) {
                    for (unsigned int i = 0; i < 3 * w * h; i++) {
                        _LOCAL_PPM_SIFT(bd[i], BYTE, ParseInt)
                    }
                } else {
                    fac = 255.0f / static_cast<float>(v);
                    for (unsigned int i = 0; i < 3 * w * h; i++) {
                        _LOCAL_PPM_SIFT(f, float, ParseDouble)
                        bd[i] = static_cast<BYTE>(f * fac);
                    }
                }

                return true;

            } else if (v <= 65535) { // word
                img.CreateImage(w, h, 3, BitmapImage::CHANNELTYPE_WORD, NULL);
                img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
                WORD *wd = img.PeekDataAs<WORD>();

                fac = 65535.0f / static_cast<float>(v);
                for (unsigned int i = 0; i < 3 * w * h; i++) {
                    _LOCAL_PPM_SIFT(f, float, ParseDouble)
                    wd[i] = static_cast<WORD>(f * fac);
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
                    _LOCAL_PPM_SIFT(f, float, ParseDouble)
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
    const BYTE *bd = img.PeekDataAs<BYTE>();
    const WORD *wd = NULL;
    SIZE_T headLen;
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
        wd = img.PeekDataAs<WORD>();

    } else if (img.GetChannelType() == BitmapImage::CHANNELTYPE_FLOAT) {
        // write a PFM

        sys::SystemInformation::Endianness machineEnd =
            sys::SystemInformation::SystemEndianness();

        if (img.GetChannelCount() != 1 && img.GetChannelCount() != 3) {
            throw new vislib::MissingImplementationException(
                "This implementation can only cope with RGB and greyscale images",
                __FILE__, __LINE__);
        }
        if (machineEnd != sys::SystemInformation::ENDIANNESS_BIG_ENDIAN
            && machineEnd != sys::SystemInformation::ENDIANNESS_LITTLE_ENDIAN) {
                throw new vislib::MissingImplementationException(
                    "Your machine is too exotic for this implementation",
                    __FILE__, __LINE__);
        }

        vislib::StringA data;
        vislib::StringA tmp;
        data.Format("P%c\n%u %u\n%f\n", img.GetChannelCount() == 1 ? 'f' : 'F', img.Width(), img.Height(), 
            machineEnd == sys::SystemInformation::ENDIANNESS_BIG_ENDIAN ? 1.0f : -1.0f);

        SIZE_T imgLen = img.GetChannelCount() * img.Width() * img.Height() * sizeof(float);
        SIZE_T bodyLen = data.Length() + imgLen;
        outmem.EnforceSize(bodyLen);
        memcpy(outmem, data.PeekBuffer(), data.Length());
        memcpy(outmem.At(data.Length()), img.PeekDataAs<UINT8>(), imgLen);

        return true;
    }

    StringA header;
    header.Format("P%d\n%u %u\n%u\n",
        bin ? 6 : 3, img.Width(), img.Height(), maxVal);
    outmem.EnforceSize(headLen = header.Length());
    memcpy(outmem, header.PeekBuffer(), headLen);

    if (bin) {
        ASSERT(bd != NULL);
        outmem.AssertSize(headLen + imgSize * 3, true);
        for (unsigned int i = 0; i < imgSize; i++) {
            outmem.AsAt<BYTE>(headLen)[i * 3]
                = (cr != UINT_MAX) ? bd[i * cc + cr] : 0;
            outmem.AsAt<BYTE>(headLen)[i * 3 + 1]
                = (cg != UINT_MAX) ? bd[i * cc + cg] : 0;
            outmem.AsAt<BYTE>(headLen)[i * 3 + 2]
                = (cb != UINT_MAX) ? bd[i * cc + cb] : 0;
        }

    } else {
        vislib::StringA data;
        vislib::StringA tmp;
        unsigned int ppl; // pixel per line

        tmp.Format("%u %u %u ", maxVal, maxVal, maxVal);
        ppl = 70 / tmp.Length();
        if (ppl < 1) ppl = 1;

        if (bd != NULL) {
            for (unsigned int i = 0; i < imgSize; i++) {
                tmp.Format("%u %u %u%c",
                    (cr != UINT_MAX) ? bd[i * cc + cr] : 0,
                    (cg != UINT_MAX) ? bd[i * cc + cg] : 0,
                    (cb != UINT_MAX) ? bd[i * cc + cb] : 0,
                    (((i + 1) % ppl) == 0) ? '\n' : ' ');
                data += tmp; // slow!
            }
        } else if (wd != NULL) {
            for (unsigned int i = 0; i < imgSize; i++) {
                tmp.Format("%u %u %u%c",
                    (cr != UINT_MAX) ? wd[i * cc + cr] : 0,
                    (cg != UINT_MAX) ? wd[i * cc + cg] : 0,
                    (cb != UINT_MAX) ? wd[i * cc + cb] : 0,
                    (((i + 1) % ppl) == 0) ? '\n' : ' ');
                data += tmp; // slow!
            }
        } else return false; // internal format error

        SIZE_T bodyLen = data.Length();
        outmem.AssertSize(headLen + bodyLen, true);
        memcpy(outmem.At(headLen), data.PeekBuffer(), bodyLen);
    }

    return true;
}


/*
 * vislib::graphics::PpmBitmapCodec::saveToMemoryImplemented
 */
bool vislib::graphics::PpmBitmapCodec::saveToMemoryImplemented(void) const {
    return true;
}
