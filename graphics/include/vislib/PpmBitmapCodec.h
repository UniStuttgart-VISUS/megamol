/*
 * PpmBitmapCodec.h
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PPMBITMAPCODEC_H_INCLUDED
#define VISLIB_PPMBITMAPCODEC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractBitmapCodec.h"


namespace vislib {
namespace graphics {


    /**
     * Bitmap codec for the portable pixmap file format (ppm).
     *
     * This is a very simple file format which is implemented mainly for
     * development purpose. This codec implements all functions, however, it is
     * not recommended to use this file format in production systems.
     *
     * This code might be used as reference when implementing additional
     * bitmap codec classes.
     */
    class PpmBitmapCodec : public AbstractBitmapCodec {
    public:

        /** Ctor. */
        PpmBitmapCodec(void);

        /** Dtor. */
        virtual ~PpmBitmapCodec(void);

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
         * Answers whether this codec can load images from memory buffers.
         *
         * @return 'true' if this codec can load images from memory buffers.
         */
        virtual bool CanLoadFromMemory(void) const;

        /**
         * Answers whether this codec can save images to memory buffers.
         *
         * @return 'true' if this codec can save images to memory buffers.
         */
        virtual bool CanSaveToMemory(void) const;

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
         * Gets the save option.
         *
         * @return 'true' if image data is saved binary (if possible).
         */
        inline bool GetSaveOption(void) const {
            return this->saveBinary;
        }

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
        virtual bool Load(const void *mem, SIZE_T size);

        /* keeping overloaded 'Load' methods */
        using AbstractBitmapCodec::Load;

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
        virtual bool Save(vislib::RawStorage& outmem) const;

        /* keeping overloaded 'Save' methods */
        using AbstractBitmapCodec::Save;

        /**
         * Sets the save option.
         *
         * @param asBinary If set to 'true' will first try to store the image
         *                 as binary, before falling back to ASCII output.
         */
        inline void SetSaveOption(bool asBinary) {
            this->saveBinary = asBinary;
        }

    private:

        /** Flag whether or not to save image data as binary */
        bool saveBinary;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PPMBITMAPCODEC_H_INCLUDED */

