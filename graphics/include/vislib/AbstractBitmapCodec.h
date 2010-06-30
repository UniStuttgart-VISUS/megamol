/*
 * AbstractBitmapCodec.h
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTBITMAPCODEC_H_INCLUDED
#define VISLIB_ABSTRACTBITMAPCODEC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/BitmapImage.h"
#include "vislib/File.h"
#include "vislib/MultiSz.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {


    /**
     * Abstract base class for all bitmap codecs implementations
     *
     * Add your new codec (derived class) to the list of built-in default
     * codecs in 'BitmapCodecCollection::BuildDefaultCollection'
     *
     */
    class AbstractBitmapCodec {
    public:

        /** Ctor. */
        AbstractBitmapCodec(void);

        /** Dtor. */
        virtual ~AbstractBitmapCodec(void);

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
         * Answers whether this codec can load images.
         *
         * @return 'true' if this codec can load images.
         */
        inline bool CanLoad(void) const {
            return this->CanLoadFromFile()
                || this->CanLoadFromMemory()
                || this->CanLoadFromStream();
        }

        /**
         * Answers whether this codec can load images from files.
         *
         * @return 'true' if this codec can load images from files.
         */
        virtual bool CanLoadFromFile(void) const;

        /**
         * Answers whether this codec can load images from memory buffers.
         *
         * @return 'true' if this codec can load images from memory buffers.
         */
        virtual bool CanLoadFromMemory(void) const;

        /**
         * Answers whether this codec can load images from file streams.
         *
         * @return 'true' if this codec can load images from file streams.
         */
        virtual bool CanLoadFromStream(void) const;

        /**
         * Answers whether this codec can save images.
         *
         * @return 'true' if this codec can save images.
         */
        inline bool CanSave(void) const {
            return this->CanSaveToFile()
                || this->CanSaveToMemory()
                || this->CanSaveToStream();
        }

        /**
         * Answers whether this codec can save images to files.
         *
         * @return 'true' if this codec can save images to files.
         */
        virtual bool CanSaveToFile(void) const;

        /**
         * Answers whether this codec can save images to memory buffers.
         *
         * @return 'true' if this codec can save images to memory buffers.
         */
        virtual bool CanSaveToMemory(void) const;

        /**
         * Answers whether this codec can save images to file streams.
         *
         * @return 'true' if this codec can save images to file streams.
         */
        virtual bool CanSaveToStream(void) const;

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
         * Accesses the pointer to the image to be used by the codec. Use this
         * to set the pointer to an image object to be used before calling
         * 'Load' or 'Save'. The image objects memory will not be released by
         * the codec object if the image pointer is changed or the codec
         * object is destroied. The caller is responsible to handling the
         * memory.
         *
         * @return Reference to the image object pointer of the codec.
         */
        inline BitmapImage* &Image(void) {
            return this->img;
        }

        /**
         * Gets the pointer to the image object that is used by the codec.
         *
         * @return The image object pointer that is used by the codec.
         */
        inline const BitmapImage *Image(void) const {
            return this->img;
        }

        /**
         * Loads an image from a file.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param filename The path to the image file to load.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool Load(const vislib::StringA& filename);

        /**
         * Loads an image from a file.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param filename The path to the image file to load.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool Load(const vislib::StringW& filename);

        /**
         * Loads an image from a file stream.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param file The file stream to load. The file stream must be opened
         *             for reading and must point to the beginning of the
         *             image data.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool Load(vislib::sys::File& file);

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

        /**
         * Loads an image from a memory buffer.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param mem The memory buffer holding the image data.
         *
         * @return 'true' if the file was successfully loaded.
         */
        virtual bool Load(const vislib::RawStorage& mem);

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const char * NameA(void) const = 0;

        /**
         * Answer the human-readable name of the codec.
         *
         * @return The human-readable name of the codec.
         */
        virtual const wchar_t * NameW(void) const = 0;

        /**
         * Saves the image to a file.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param filename The path to the image file to be written.
         * @param overwrite Flag whether or not existing files should be
         *                  overwritten. If 'true' existing files will be
         *                  overwritten, if 'false' the method will fail if
         *                  the file already exists.
         *
         * @return 'true' if the file was successfully saved.
         */
        virtual bool Save(const vislib::StringA& filename,
            bool overwrite = true) const;

        /**
         * Saves the image to a file.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param filename The path to the image file to be written.
         * @param overwrite Flag whether or not existing files should be
         *                  overwritten. If 'true' existing files will be
         *                  overwritten, if 'false' the method will fail if
         *                  the file already exists.
         *
         * @return 'true' if the file was successfully saved.
         */
        virtual bool Save(const vislib::StringW& filename,
            bool overwrite = true) const;

        /**
         * Saves the image to a file stream. The method will not close the
         * stream. The image data will be written beginning from the current
         * writing position of the stream. The streams writing position will
         * be placed after the image data.
         *
         * You must set 'Image' to a valid BitmapImage object before calling
         * this method.
         *
         * @param file The file stream to store the image in. The stream must
         *             be opened for writing.
         *
         * @return 'true' if the file was successfully saved.
         */
        virtual bool Save(vislib::sys::File& file) const;

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

    protected:

        /**
         * Gets the image to be used by the codec.
         *
         * @return The image to be used.
         *
         * @throw IllegalStateException if no image is set
         */
        BitmapImage& image(void);

        /**
         * Gets the image to be used by the codec.
         *
         * @return The image to be used.
         *
         * @throw IllegalStateException if no image is set
         */
        const BitmapImage& image(void) const;

    private:

        /** The bitmap image used by the codec */
        BitmapImage *img;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTBITMAPCODEC_H_INCLUDED */

