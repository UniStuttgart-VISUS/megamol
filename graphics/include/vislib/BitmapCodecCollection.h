/*
 * BitmapCodecCollection.h
 *
 * Copyright (C) 2010 by SGrottel
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BITMAPCODECCOLLECTION_H_INCLUDED
#define VISLIB_BITMAPCODECCOLLECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractBitmapCodec.h"
#include "vislib/Array.h"
#include "vislib/BitmapImage.h"
#include "vislib/File.h"
#include "vislib/RawStorage.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"


namespace vislib {
namespace graphics {


    /**
     * Bitmap codec collections are used to automatically choose the codec
     * for a file to load as a BitmapImage
     */
    class BitmapCodecCollection {
    public:

        /** Array of codecs */
        typedef Array<SmartPtr<AbstractBitmapCodec> > CodecArray;

        /**
         * Builds a new collection with the built-in codecs
         *
         * @return A new collection with the built-in codecs
         */
        static BitmapCodecCollection BuildDefaultCollection();

        /**
         * Answer the built-in default collection
         *
         * @return The built-in default codec collection
         */
        static BitmapCodecCollection& DefaultCollection();

        /** Ctor. */
        BitmapCodecCollection(void);

        /**
         * Copy Ctor.
         *
         * @param src The object to clone from
         */
        BitmapCodecCollection(const BitmapCodecCollection& src);

        /** Dtor. */
        ~BitmapCodecCollection(void);

        /**
         * Adds a codec to the collection. If this codec already is in the
         * collection it will not be added.
         *
         * @param codec The codec to be added.
         */
        inline void AddCodec(const SmartPtr<AbstractBitmapCodec>& codec) {
            if (!this->codecs.Contains(codec)) {
                this->codecs.Add(codec);
            }
        }

        /**
         * Adds a codec to the collection. If this codec already is in the
         * collection it will not be added.
         *
         * @param codec The codec to be added.
         */
        inline void AddCodec(AbstractBitmapCodec *codec) {
            this->AddCodec(SmartPtr<AbstractBitmapCodec>(codec));
        }

        /**
         * Removes all codecs from the collection
         */
        inline void Clear(void) {
            this->codecs.Clear();
        }

        /**
         * Answer the idx-th codec of the collection
         *
         * @param idx The zero-based index of the codec to return
         *
         * @return The idx-th codec
         */
        inline SmartPtr<AbstractBitmapCodec> Codec(SIZE_T idx) {
            return this->codecs[idx];
        }

        /**
         * Answer the idx-th codec of the collection
         *
         * @param idx The zero-based index of the codec to return
         *
         * @return The idx-th codec
         */
        inline const SmartPtr<AbstractBitmapCodec> Codec(SIZE_T idx) const {
            return this->codecs[idx];
        }

        /**
         * Answer the number of codecs in the collection
         *
         * @return The number of codecs in the collection
         */
        inline SIZE_T Count(void) const {
            return this->codecs.Count();
        }

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param filename The path to the file to load
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        bool LoadBitmapImage(BitmapImage& outImg,
            const vislib::StringA& filename);

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param filename The path to the file to load
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        bool LoadBitmapImage(BitmapImage& outImg,
            const vislib::StringW& filename);

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param filename The path to the file to load
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        inline bool LoadBitmapImage(BitmapImage& outImg,
                const char *filename) {
            return this->LoadBitmapImage(outImg, vislib::StringA(filename));
        }

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param filename The path to the file to load
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        inline bool LoadBitmapImage(BitmapImage& outImg,
                const wchar_t *filename) {
            return this->LoadBitmapImage(outImg, vislib::StringW(filename));
        }

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param file The file stream to load the image from (current location)
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        bool LoadBitmapImage(BitmapImage& outImg, vislib::sys::File& file);

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param mem The memory to load the image from
         * @param size The size of the memory in bytes
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        bool LoadBitmapImage(BitmapImage& outImg,
            const void *mem, SIZE_T size);

        /**
         * Loads a bitmap image by automatically choosing the proper codec
         *
         * @param outImg The BitmapImage to receive the loaded image
         * @param mem The memory to load the image from
         *
         * @return True on success
         *
         * @throw vislib::Exception or derived class on error
         */
        inline bool LoadBitmapImage(BitmapImage& outImg,
                const vislib::RawStorage& mem) {
            return this->LoadBitmapImage(outImg, mem, mem.GetSize());
        }

        /**
         * Removes a specific codec from the collection
         *
         * @param codec The codec to be removed from the collection
         */
        inline void RemoveCodec(const SmartPtr<AbstractBitmapCodec> codec) {
            this->codecs.RemoveAll(codec);
        }

        /**
         * Removes the idx-th codec from the collection
         *
         * @param idx The zero-based index of the codec to be removed
         */
        inline void RemoveCodec(SIZE_T idx) {
            this->codecs.RemoveAt(idx);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline BitmapCodecCollection& operator=(const BitmapCodecCollection& rhs) {
            this->codecs = rhs.codecs;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this and rhs are equal
         */
        inline bool operator==(const BitmapCodecCollection& rhs) const {
            return this->codecs == rhs.codecs;
        }

        /**
         * Answer the idx-th codec of the collection
         *
         * @param idx The zero-based index of the codec to return
         *
         * @return The idx-th codec
         */
        inline SmartPtr<AbstractBitmapCodec>& operator[](SIZE_T idx) {
            return this->codecs[idx];
        }

        /**
         * Answer the idx-th codec of the collection
         *
         * @param idx The zero-based index of the codec to return
         *
         * @return The idx-th codec
         */
        inline const SmartPtr<AbstractBitmapCodec>& operator[](SIZE_T idx) const {
            return this->codecs[idx];
        }

    private:

        /**
         * Selects the codecs matching the file name extensions
         *
         * @param filename The file name
         * @param outCodecs The codec array receiving the matching codecs
         */
        void selectCodecsByFilename(const vislib::StringA& filename,
            CodecArray& outCodecs) const;

        /**
         * Selects the codecs matching the file name extensions
         *
         * @param filename The file name
         * @param outCodecs The codec array receiving the matching codecs
         */
        void selectCodecsByFilename(const vislib::StringW& filename,
            CodecArray& outCodecs) const;

        /**
         * Performs codec auto detection
         *
         * @param mem The preview data
         * @param size The preview data size
         * @param codecs The codecs to test
         * @param outMatchingCodecs The codecs out of 'codecs' which
         *                          successfully autodetected (return value 1)
         * @param outUnsureCodecs The codecs out of 'codecs' which may be able
         *                        to load the image (return value -1)
         */
        void autodetecCodec(const void *mem, SIZE_T size,
            const CodecArray& codecs, CodecArray& outMatchingCodecs,
            CodecArray& outUnsureCodecs) const;

        /** The codecs in this collection */
        CodecArray codecs;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BITMAPCODECCOLLECTION_H_INCLUDED */

