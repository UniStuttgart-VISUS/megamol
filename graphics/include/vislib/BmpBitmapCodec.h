/*
 * BmpBitmapCodec.h
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BMPBITMAPCODEC_H_INCLUDED
#define VISLIB_BMPBITMAPCODEC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractBitmapCodec.h"
#include "vislib/types.h"

//
// Use this define to load Bitmaps manually on windows if the windows api
// functions fail, or for debugging
//
//#define VISLIB_BMP_LOAD_BY_HAND 1


namespace vislib {
namespace graphics {


    /**
     * Bitmap codec for the windows bitmap file format.
     * Only uncompressed bitmaps are supported (no rle, png, jpg).
     */
    class BmpBitmapCodec : public AbstractBitmapCodec {
    public:

        /** Ctor. */
        BmpBitmapCodec(void);

        /** Dtor. */
        virtual ~BmpBitmapCodec(void);

        /**
         * Autodetects if an image can be loaded by this codec by checking
         * preview data from the beginning of the image data.
         *
         * @param mem The preview data.
         * @param size The size of the preview data in bytes.
         *
         * @return 0 if the file cannot be loaded by this codec.
         *         -1 if the preview data was insufficient to determine the
         *            codec compatibility.
         *         1 if the file can be loaded by this codec (loading might
         *           still fail however, e.g. if file data is corrupt).
         */
        virtual int AutoDetect(const void *mem, SIZE_T size) const;

        /**
         * Answers whether this codec can autodetect if an image is supported
         * by checking preview data.
         *
         * @return 'true' if the codec can autodetect image compatibility.
         */
        virtual bool CanAutoDetect(void) const;

        /**
         * Answer the file name extensions usually used for image files of
         * the type of this codec. Each file name extension includes the
         * leading period. Multiple file name extensions are separated by
         * semicolons.
         *
         * @return The file name extensions usually used for image files of
         *         the type of this codec.
         */
        virtual const char* FileNameExtsA(void) const;

        /**
         * Answer the file name extensions usually used for image files of
         * the type of this codec. Each file name extension includes the
         * leading period. Multiple file name extensions are separated by
         * semicolons.
         *
         * @return The file name extensions usually used for image files of
         *         the type of this codec.
         */
        virtual const wchar_t* FileNameExtsW(void) const;

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const char * NameA(void) const;

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const wchar_t * NameW(void) const;

    protected:

        /**
         * Loads an image from a memory buffer.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param mem Pointer to the memory buffer holding the image data.
         * @param size The size of the memory buffer in bytes.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool loadFromMemory(const void *mem, SIZE_T size);

        /**
         * Answer whether or not 'loadFromMemory' has been implement.
         *
         * @return true
         */
        virtual bool loadFromMemoryImplemented(void) const;

        /**
         * Saves the image to a memory block.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param outmem The memory block to receive the image data. The image
         *               data will replace all data in the memory block.
         *
         * @return 'true' if the file was successfully saved.
         */
        virtual bool saveToMemory(vislib::RawStorage& outmem) const;

        /**
         * Answer whether or not 'saveToMemory' has been implement.
         *
         * @return true
         */
        virtual bool saveToMemoryImplemented(void) const;

    private:

#if defined(VISLIB_BMP_LOAD_BY_HAND) || !defined(_WIN32)

        /**
         * Loads the remaining bitmap data, starting with a BITMAPINFOHEADER
         * structure.
         *
         * @param header The already read BITMAPFILEHEADER
         * @param dat Pointer to the bitmap data
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadWithBitmapInfoHeader(const void *header, const BYTE *dat);

        /**
         * Loads a 1 bit bitmap without palette
         *
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param stride The number of bytes per scan line
         * @param colPalDat The colour palette
         * @param colPalSize The number of entries in the colour palette
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap1(int width, int height, int stride,
            const void *colPalDat, unsigned int colPalSize, const BYTE *dat);

        /**
         * Loads a 4 bit bitmap without palette
         *
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param stride The number of bytes per scan line
         * @param colPalDat The colour palette
         * @param colPalSize The number of entries in the colour palette
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap4(int width, int height, int stride,
            const void *colPalDat, unsigned int colPalSize, const BYTE *dat);

        /**
         * Loads a 8 bit bitmap without palette
         *
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param stride The number of bytes per scan line
         * @param colPalDat The colour palette
         * @param colPalSize The number of entries in the colour palette
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap8(int width, int height, int stride,
            const void *colPalDat, unsigned int colPalSize, const BYTE *dat);

        /**
         * Loads a 16 bit bitmap without palette
         *
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param stride The number of bytes per scan line
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap16(int width, int height, int stride, const BYTE *dat);

        /**
         * Loads a 24 bit bitmap without palette
         *
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param stride The number of bytes per scan line
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap24(int width, int height, int stride, const BYTE *dat);

        /**
         * Loads a 32 bit bitmap without palette
         * @param width The number of pixels in one scan line
         * @param height The number of scan lines. If negative the image is
         *               stored vertically flipped.
         * @param dat Pointer to the first byte of the first scan line
         *
         * @return 'true' on success, 'false' on failure
         */
        bool loadBitmap32(int width, int height, const BYTE *dat);

#endif /* defined(VISLIB_BMP_LOAD_BY_HAND) || !defined(_WIN32) */

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BMPBITMAPCODEC_H_INCLUDED */

