/*
 * PngBitmapCodec.h
 *
 * Copyright (C) 2010 by Sebastian Grottel.
 * (Copyright (C) 2010 by VISUS (Universitaet Stuttgart))
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PNGBITMAPCODEC_H_INCLUDED
#define VISLIB_PNGBITMAPCODEC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/RawStorage.h"
#include "vislib/graphics/AbstractBitmapCodec.h"


namespace sg {
namespace graphics {

/**
 * Bitmap codec for png images
 * Currently loading only
 */
class PngBitmapCodec
#ifdef _WIN32
#pragma warning(disable : 4275)
#endif /* _WIN32 */
        : public vislib::graphics::AbstractBitmapCodec {
#ifdef _WIN32
#pragma warning(default : 4275)
#endif /* _WIN32 */
public:
    /** Ctor */
    PngBitmapCodec();

    /** Dtor */
    ~PngBitmapCodec() override;

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
     * Loads the image from a block of memory
     *
     * @param mem The block of memory
     * @param size The size of the block of memory
     *
     * @return true on success, false on failure
     */
    bool loadFromMemory(const void* mem, SIZE_T size) override;

    /**
     * Answer whether or not 'loadFromMemory' has been implement.
     *
     * @return true
     */
    bool loadFromMemoryImplemented() const override;

    /**
     * Saves the image to a file stream
     *
     * @param stream The file stream
     *
     * @return true on success, false on failure
     */
    bool saveToStream(vislib::sys::File& stream) const override;

    /**
     * Answer whether or not 'saveToStream' has been implement.
     *
     * The default implementation returns 'false'. Overwrite to return
     * 'true' when you implement 'saveToStream' in a derived class.
     *
     * @return true if 'saveToStream' has been implemented
     */
    bool saveToStreamImplemented() const override;
};


} /* end namespace graphics */
} /* end namespace sg */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PNGBITMAPCODEC_H_INCLUDED */
