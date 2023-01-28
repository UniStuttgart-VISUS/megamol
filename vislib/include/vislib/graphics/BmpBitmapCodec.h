/*
 * BmpBitmapCodec.h
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BMPBITMAPCODEC_H_INCLUDED
#define VISLIB_BMPBITMAPCODEC_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/graphics/AbstractBitmapCodec.h"
#include "vislib/types.h"

//
// Use this define to load Bitmaps manually on windows if the windows api
// functions fail, or for debugging
//
//#define VISLIB_BMP_LOAD_BY_HAND 1


namespace vislib::graphics {


/**
 * Bitmap codec for the windows bitmap file format.
 * Only uncompressed bitmaps are supported (no rle, png, jpg).
 */
class BmpBitmapCodec : public AbstractBitmapCodec {
public:
    /** Ctor. */
    BmpBitmapCodec();

    /** Dtor. */
    ~BmpBitmapCodec() override;

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
    int AutoDetect(const void* mem, SIZE_T size) const override;

    /**
     * Answers whether this codec can autodetect if an image is supported
     * by checking preview data.
     *
     * @return 'true' if the codec can autodetect image compatibility.
     */
    bool CanAutoDetect() const override;

    /**
     * Answer the file name extensions usually used for image files of
     * the type of this codec. Each file name extension includes the
     * leading period. Multiple file name extensions are separated by
     * semicolons.
     *
     * @return The file name extensions usually used for image files of
     *         the type of this codec.
     */
    const char* FileNameExtsA() const override;

    /**
     * Answer the file name extensions usually used for image files of
     * the type of this codec. Each file name extension includes the
     * leading period. Multiple file name extensions are separated by
     * semicolons.
     *
     * @return The file name extensions usually used for image files of
     *         the type of this codec.
     */
    const wchar_t* FileNameExtsW() const override;

    /**
     * Answer the human-readable name of the codec.
     *
     * @return The human-readable name of the codec.
     */
    const char* NameA() const override;

    /**
     * Answer the human-readable name of the codec.
     *
     * @return The human-readable name of the codec.
     */
    const wchar_t* NameW() const override;

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
    bool loadFromMemory(const void* mem, SIZE_T size) override;

    /**
     * Answer whether or not 'loadFromMemory' has been implement.
     *
     * @return true
     */
    bool loadFromMemoryImplemented() const override;

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
    bool saveToMemory(vislib::RawStorage& outmem) const override;

    /**
     * Answer whether or not 'saveToMemory' has been implement.
     *
     * @return true
     */
    bool saveToMemoryImplemented() const override;

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
    bool loadWithBitmapInfoHeader(const void* header, const BYTE* dat);

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
    bool loadBitmap1(
        int width, int height, int stride, const void* colPalDat, unsigned int colPalSize, const BYTE* dat);

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
    bool loadBitmap4(
        int width, int height, int stride, const void* colPalDat, unsigned int colPalSize, const BYTE* dat);

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
    bool loadBitmap8(
        int width, int height, int stride, const void* colPalDat, unsigned int colPalSize, const BYTE* dat);

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
    bool loadBitmap16(int width, int height, int stride, const BYTE* dat);

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
    bool loadBitmap24(int width, int height, int stride, const BYTE* dat);

    /**
     * Loads a 32 bit bitmap without palette
     * @param width The number of pixels in one scan line
     * @param height The number of scan lines. If negative the image is
     *               stored vertically flipped.
     * @param dat Pointer to the first byte of the first scan line
     *
     * @return 'true' on success, 'false' on failure
     */
    bool loadBitmap32(int width, int height, const BYTE* dat);

#endif /* defined(VISLIB_BMP_LOAD_BY_HAND) || !defined(_WIN32) */
};

} // namespace vislib::graphics

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BMPBITMAPCODEC_H_INCLUDED */
