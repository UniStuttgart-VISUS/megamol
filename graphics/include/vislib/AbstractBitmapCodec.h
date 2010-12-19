/*
 * AbstractBitmapCodec.h
 *
 * Copyright (C) 2009 - 2010 by Sebastian Grottel.
 * (Copyright (C) 2009 - 2010 by VISUS (Universität Stuttgart))
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
     * When implementing a codec for loading, you should at least implement
     * 'loadFromStream' and 'loadFromStreamImplemented'.
     * All other 'loadFrom*' Methods can be implemented for better performace.
     *
     * When implementing a codec for saving, you should at least implement
     * 'saveToStream' and 'saveToStreamImplemented'.
     * All other 'saveTo*' Methods can be implemented for better performance.
     *
     * Add your new codec (derived class) to the list of built-in default
     * codecs in 'BitmapCodecCollection::BuildDefaultCollection'
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
            return this->loadFromFileAImplemented()
                || this->loadFromFileWImplemented()
                || this->loadFromMemoryImplemented()
                || this->loadFromStreamImplemented();
        }

        /**
         * Answers whether this codec can save images.
         *
         * @return 'true' if this codec can save images.
         */
        inline bool CanSave(void) const {
            return this->saveToFileAImplemented()
                || this->saveToFileWImplemented()
                || this->saveToMemoryImplemented()
                || this->saveToStreamImplemented();
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
        bool Load(const char* filename);

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
        inline bool Load(const vislib::StringA& filename) {
            return this->Load(filename.PeekBuffer());
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
        bool Load(const wchar_t* filename);

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
        inline bool Load(const vislib::StringW& filename) {
            return this->Load(filename.PeekBuffer());
        }

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
        bool Load(vislib::sys::File& file);

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
        bool Load(const void *mem, SIZE_T size);

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
        inline bool Load(const vislib::RawStorage& mem) {
            return this->Load(mem, mem.GetSize());
        }

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
        bool Save(const char* filename, bool overwrite = true) const;

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
        inline bool Save(const vislib::StringA& filename,
                bool overwrite = true) const {
            return this->Save(filename.PeekBuffer(), overwrite);
        }

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
        bool Save(const wchar_t* filename, bool overwrite = true) const;

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
        inline bool Save(const vislib::StringW& filename,
                bool overwrite = true) const {
            return this->Save(filename.PeekBuffer(), overwrite);
        }

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
        bool Save(vislib::sys::File& file) const;

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
        bool Save(vislib::RawStorage& outmem) const;

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

        /**
         * Loads the image from a file
         *
         * @param filename The path to the image file to load
         *
         * @return true on success, false on failure
         */
        virtual bool loadFromFileA(const char *filename);

        /**
         * Answer whether or not 'loadFromFileA' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'loadFromFileA' in a derived class.
         *
         * @return true if 'loadFromFileA' has been implemented
         */
        virtual bool loadFromFileAImplemented(void) const;

        /**
         * Loads the image from a file
         *
         * @param filename The path to the image file to load
         *
         * @return true on success, false on failure
         */
        virtual bool loadFromFileW(const wchar_t *filename);

        /**
         * Answer whether or not 'loadFromFileW' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'loadFromFileW' in a derived class.
         *
         * @return true if 'loadFromFileW' has been implemented
         */
        virtual bool loadFromFileWImplemented(void) const;

        /**
         * Loads the image from a block of memory
         *
         * @param mem The block of memory
         * @param size The size of the block of memory
         *
         * @return true on success, false on failure
         */
        virtual bool loadFromMemory(const void *mem, SIZE_T size);

        /**
         * Answer whether or not 'loadFromMemory' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'loadFromMemory' in a derived class.
         *
         * @return true if 'loadFromMemory' has been implemented
         */
        virtual bool loadFromMemoryImplemented(void) const;

        /**
         * Loads the image from a file stream
         *
         * @param stream The file stream
         *
         * @return true on success, false on failure
         */
        virtual bool loadFromStream(vislib::sys::File& stream);

        /**
         * Answer whether or not 'loadFromStream' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'loadFromStream' in a derived class.
         *
         * @return true if 'loadFromStream' has been implemented
         */
        virtual bool loadFromStreamImplemented(void) const;

        /**
         * Saves the image to a file
         *
         * @param filename The path to the file
         *
         * @return true on success, false on failure
         */
        virtual bool saveToFileA(const char *filename) const;

        /**
         * Answer whether or not 'saveToFileA' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'saveToFileA' in a derived class.
         *
         * @return true if 'saveToFileA' has been implemented
         */
        virtual bool saveToFileAImplemented(void) const;

        /**
         * Saves the image to a file
         *
         * @param filename The path to the file
         *
         * @return true on success, false on failure
         */
        virtual bool saveToFileW(const wchar_t *filename) const;

        /**
         * Answer whether or not 'saveToFileW' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'saveToFileW' in a derived class.
         *
         * @return true if 'saveToFileW' has been implemented
         */
        virtual bool saveToFileWImplemented(void) const;

        /**
         * Saves the image to a block of memory
         *
         * @param mem The raw block of memory to receive the encoded image
         *
         * @return true on success, false on failure
         */
        virtual bool saveToMemory(vislib::RawStorage &mem) const;

        /**
         * Answer whether or not 'saveToMemory' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'saveToMemory' in a derived class.
         *
         * @return true if 'saveToMemory' has been implemented
         */
        virtual bool saveToMemoryImplemented(void) const;

        /**
         * Saves the image to a file stream
         *
         * @param stream The file stream
         *
         * @return true on success, false on failure
         */
        virtual bool saveToStream(vislib::sys::File& stream) const;

        /**
         * Answer whether or not 'saveToStream' has been implement.
         *
         * The default implementation returns 'false'. Overwrite to return
         * 'true' when you implement 'saveToStream' in a derived class.
         *
         * @return true if 'saveToStream' has been implemented
         */
        virtual bool saveToStreamImplemented(void) const;

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

