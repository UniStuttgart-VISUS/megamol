/*
 * BitmapPainter.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/BitmapPainter.h"
#include "vislib/assert.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"


/*
 * vislib::graphics::BitmapPainter::BitmapPainter
 */
vislib::graphics::BitmapPainter::BitmapPainter(BitmapImage *img) : img(img),
        col(), colSize(0), colBits(NULL), colMask(NULL) {
    // Intentionally empty
}


/*
 * vislib::graphics::BitmapPainter::~BitmapPainter
 */
vislib::graphics::BitmapPainter::~BitmapPainter(void) {
    this->img = NULL; // DO NOT DELETE
    this->clearColourCache();
}


/*
 * vislib::graphics::BitmapPainter::Clear
 */
void vislib::graphics::BitmapPainter::Clear(void) {
    this->preDraw();
    unsigned int bpp = this->img->BytesPerPixel();
    unsigned char *bytes = this->img->PeekDataAs<unsigned char>();
    for (unsigned int p = this->img->Width() * this->img->Height(); p > 0;
            p--, bytes += bpp) {
        this->setPixel(bytes);
    }
}


/*
 * vislib::graphics::BitmapPainter::DrawLine
 */
void vislib::graphics::BitmapPainter::DrawLine(
        int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    int sdx = vislib::math::Signum<int>(dx);
    int sdy = vislib::math::Signum<int>(dy);
    int pdx, pdy, es, el;
    //unsigned int pixelCount = 0;
    dx = vislib::math::Abs<int>(dx);
    dy = vislib::math::Abs<int>(dy);

    if (dx > dy) {
        pdx = sdx; pdy = 0;
        es = dy; el = dx;
    } else {
        pdx = 0; pdy = sdy;
        es = dx; el = dy;
    }

    int x = x1, y = y1;
    this->SetPixel(x, y);
    //pixelCount++;
    int err = el / 2;
    for (int t = 0; t < el; t++) {
        err -= es;
        if (err < 0) {
            err += el;
            x += sdx;
            y += sdy;
        } else {
            x += pdx;
            y += pdy;
        }
        this->SetPixel(x, y);
        //pixelCount++;
    }
}


/*
 * vislib::graphics::BitmapPainter::preDraw
 */
#ifdef _WIN32
inline
#endif /* _WIN32 */
void vislib::graphics::BitmapPainter::preDraw(void) {
    if (this->img == NULL) {
        throw vislib::IllegalStateException(
            "You must set an image to draw to", __FILE__, __LINE__);
    }
    if (this->colBits == NULL) {
        // rebuild colour cache
        const unsigned int unsetMask = 0x00000000;
        const unsigned int setMask = 0xFFFFFFFF;
        unsigned int cc = this->img->GetChannelCount();
        ASSERT((this->img->BytesPerPixel() % cc) == 0);
        unsigned int bpc = this->img->BytesPerPixel() / cc;
        ASSERT(bpc <= 4);
        this->colSize = this->img->BytesPerPixel();
        this->colBits = new unsigned char[this->colSize];
        this->colMask = new unsigned char[this->colSize];
        for (unsigned int ci = 0; ci < cc; ci++) {
            //unsigned char *bits = this->colBits + (ci * bpc);
            bool setValue = false;
            switch (this->img->GetChannelType()) {
                case BitmapImage::CHANNELTYPE_BYTE:
                    setValue = this->setColourCacheValue(
                        this->colBits + (ci * bpc),
                        ci, this->img->GetChannelLabel(ci));
                    break;
                case BitmapImage::CHANNELTYPE_WORD:
                    setValue = this->setColourCacheValue(
                        reinterpret_cast<unsigned short*>(
                            this->colBits + (ci * bpc)),
                        ci, this->img->GetChannelLabel(ci));
                    break;
                case BitmapImage::CHANNELTYPE_FLOAT:
                    setValue = this->setColourCacheValue(
                        reinterpret_cast<float*>(this->colBits + (ci * bpc)),
                        ci, this->img->GetChannelLabel(ci));
                    break;
            }
            ::memcpy(this->colMask + (ci * bpc),
                setValue ? &setMask : &unsetMask, bpc);
        }
#if defined(DEBUG) || defined(_DEBUG)
        for (unsigned int i = 0; i < this->colSize; i++) {
            ASSERT((this->colBits[i] & ~this->colMask[i]) == 0);
        }
#endif /* DEBUG || _DEBUG */
    }
}


/*
 * vislib::graphics::BitmapPainter::setColourCacheValue
 */
template<class Tp>
inline bool vislib::graphics::BitmapPainter::setColourCacheValue(Tp* dst,
        unsigned int idx, BitmapImage::ChannelLabel label) {
    bool rv = false;
    for (SIZE_T i = 0; i < this->col.Count(); i++) {
        ChannelColour &cc = this->col[i];
        if ((cc.idx == idx) || (cc.label == label)
                || ((cc.idx == UINT_MAX)
                && (cc.label == BitmapImage::CHANNEL_UNDEF))) {
            switch (cc.type) {
                case BitmapImage::CHANNELTYPE_BYTE:
                    this->setColourCacheValue(dst, cc.value.asByte);
                    break;
                case BitmapImage::CHANNELTYPE_WORD:
                    this->setColourCacheValue(dst, cc.value.asWord);
                    break;
                case BitmapImage::CHANNELTYPE_FLOAT:
                    this->setColourCacheValue(dst, cc.value.asFloat);
                    break;
            }
            rv = true;
        }
    }
    return rv;
}


/*
 * vislib::graphics::BitmapPainter::setPixel
 */
#ifdef _WIN32
VISLIB_FORCEINLINE
#endif /* _WIN32 */
void vislib::graphics::BitmapPainter::setPixel(
        unsigned char *dst) {
    for (unsigned int i = 0; i < this->colSize; i++) {
        dst[i] = (this->colBits[i] & this->colMask[i])
            | (dst[i] & ~this->colMask[i]);
    }
}
