/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "vislib/RawStorage.h"
#include "vislib/graphics/AbstractBitmapCodec.h"
#ifdef _WIN32
#include <Shlwapi.h>
#include <Wincodecsdk.h>
#include <windows.h>
#endif


namespace sg {
namespace graphics {

/**
 * Bitmap codec for jpeg images using the IJG libjpeg
 * Currently loading only
 */
class JpegBitmapCodec : public vislib::graphics::AbstractBitmapCodec {
public:
    /** Ctor */
    JpegBitmapCodec(void);

    /** Dtor */
    virtual ~JpegBitmapCodec(void);

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
    virtual int AutoDetect(const void* mem, SIZE_T size) const;

    /**
     * Answers whether this codec can autodetect if an image is supported
     * by checking preview data.
     *
     * @return 'true' if the codec can autodetect image compatibility.
     */
    virtual bool CanAutoDetect(void) const;

    /**
     * Answer the compression quality setting, that will be used when
     * encoding the image.
     *
     * @return The compression quality setting [0..100]
     */
    inline unsigned int CompressionQuality(void) const {
        return this->quality;
    }

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
    virtual const char* NameA(void) const;

    /**
     * Answer the human-readable name of the codec.
     *
     * @return The human-readable name of the codec.
     */
    virtual const wchar_t* NameW(void) const;

    /**
     * Analysis the set image and automatically evaluates the optimal
     * compression quality for a high image quality.
     *
     * Warning: This method is extremely slow, especially for large
     *    images! It might by a good idea to use the method on smaller
     *    sub-images.
     */
    void OptimizeCompressionQuality(void);

    /**
     * Sets the compression quality. Values will be clamped to [0..100].
     * Larger values result in higher quality and large file sizes.
     *
     * @param q The new value for the compression quality
     */
    inline void SetCompressionQuality(unsigned int q) {
        this->quality = (q < 100) ? q : 100;
    }

protected:
    /**
     * Loads the image from a block of memory
     *
     * @param mem The block of memory
     * @param size The size of the block of memory
     *
     * @return true on success, false on failure
     */
    virtual bool loadFromMemory(const void* mem, SIZE_T size);

    /**
     * Answer whether or not 'loadFromMemory' has been implement.
     *
     * @return true
     */
    virtual bool loadFromMemoryImplemented(void) const;

    /**
     * Saves the image to a block of memory
     *
     * @param mem The raw block of memory to receive the encoded image
     *
     * @return true on success, false on failure
     */
    virtual bool saveToMemory(vislib::RawStorage& mem) const;

    /**
     * Answer whether or not 'saveToMemory' has been implement.
     *
     * The default implementation returns 'false'. Overwrite to return
     * 'true' when you implement 'saveToMemory' in a derived class.
     *
     * @return true if 'saveToMemory' has been implemented
     */
    virtual bool saveToMemoryImplemented(void) const;

    virtual bool saveToStream(vislib::sys::File& stream) const;

    virtual bool saveToStreamImplemented(void) const;

private:
    /** The compression quality setting [0..100] */
    unsigned int quality;

#ifdef _WIN32
    IWICImagingFactory* piFactory = NULL;
    bool comOK;
#endif
};


} /* end namespace graphics */
} /* end namespace sg */
