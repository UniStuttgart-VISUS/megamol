/*
 * RawStorageWriter.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RAWSTORAGEWRITER_H_INCLUDED
#define VISLIB_RAWSTORAGEWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "the/types.h"
#include "vislib/RawStorage.h"
#include "vislib/String.h"
#include <climits>


namespace vislib {


    /**
     * Writer utility class managing a RawStorage allowing for bigger
     * memory appends for small data blocks and thus better performance.
     */
    class RawStorageWriter {
    public:

        /** The default memory increment value */
        static const size_t DEFAULT_INCREMENT = 4 * 1024;

        /**
         * Ctor.
         *
         * @param store A reference to the RawStorage object which will be used
         * @param pos The value for the position pointer
         * @param end The value for the end-of-data pointer. If this is
         *            SIZE_MAX the current size of 'store' is used. If this is
         *            less then the 'pos', it is set to the value of 'pos'.
         * @param inc The memory increment value. If this is zero, the default
         *            increment value is used.
         */
        RawStorageWriter(vislib::RawStorage &store, size_t pos = 0,
            size_t end = SIZE_MAX, size_t inc = DEFAULT_INCREMENT);

        /** Dtor. */
        ~RawStorageWriter(void);

        /**
         * Answer the reference to the raw data store
         *
         * @return The reference to the raw data store
         */
        inline vislib::RawStorage& Data(void) const {
            return this->data;
        }

        /**
         * Answer the current end-of-data pointer
         *
         * @return The end-of-data pointer
         */
        inline size_t End(void) const {
            return this->end;
        }

        /**
         * Answer the memory increment step size
         *
         * @return The memory increment step size
         */
        inline size_t Increment(void) const {
            return this->inc;
        }

        /**
         * Answer the current position pointer
         *
         * @return The position pointer
         */
        inline size_t Position(void) const {
            return this->pos;
        }

        /**
         * Sets the end-of-data pointer. If the new value is less then the
         * position pointer the position pointer will also be set to this
         * value. If the new value is more then the size of the RawStorage the
         * size of the RawStorage will be increased by multiples of the memory
         * increment step size.
         *
         * @param end The new value for the end-of-data pointer
         */
        void SetEnd(size_t end);

        /**
         * Sets the memory increment step size.
         *
         * @param inc The new memory increment step size; If this is zero, the
         *            default increment value is used.
         */
        void SetIncrement(size_t inc);

        /**
         * Sets the position pointer. If the new value is more then the
         * end-of-data pointer the end-of-data pointer is also set to this
         * value. If the new value is more then the size of the RawStorage the
         * size of the RawStorage will be increased by multiples of the memory
         * increment step size.
         *
         * @param pos The new value for the position pointer.
         */
        void SetPosition(size_t pos);

        /**
         * Writes 'size' bytes from 'buf' into the RawStorage at the current
         * position, adjustes the end-of-data pointer if required, and
         * increases the size of the RawStorage by multiples of the memory
         * increment step size if required.
         *
         * @param buf Pointer to the data to be written.
         * @param size The number of bytes to be written.
         */
        void Write(const void *buf, size_t size);

        /**
         * TODO: Document
         */
        // notice there is no 'int' or 'size_t' since these differ between plattforms
        inline void Write(uint8_t d) { this->Write(&d, 1); }
        inline void Write(int8_t d) { this->Write(&d, 1); }
        inline void Write(uint16_t d) { this->Write(&d, 2); }
        inline void Write(int16_t d) { this->Write(&d, 2); }
        inline void Write(uint32_t d) { this->Write(&d, 4); }
        inline void Write(int32_t d) { this->Write(&d, 4); }
        inline void Write(uint64_t d) { this->Write(&d, 8); }
        inline void Write(int64_t d) { this->Write(&d, 8); }
        inline void Write(float d) { this->Write(&d, 4); }
        inline void Write(double d) { this->Write(&d, 8); }
        inline void Write(const vislib::StringA& d) {
            this->Write(static_cast<unsigned int>(d.Length()));
            this->Write(d.PeekBuffer(), d.Length() * sizeof(char));
        }
        // You really do not want to use unicode, since linux has f**ked it up
        //inline Write(const vislib::StringW& d) {
        //    this->Write(static_cast<uint64_t>(d.Length()));
        //    this->Write(d.PeekBuffer(), d.Length() * sizeof(wchar_t));
        //}
        inline void Write(const char *d) {
            size_t l = vislib::CharTraitsA::SafeStringLength(d);
            this->Write(static_cast<unsigned int>(l));
            this->Write(d, l);
        }
        // You really do not want to use unicode, since linux has f**ked it up
        //inline Write(const wchar_t *d) {
        //    uint64_t l = static_cast<uint64_t>(vislib::CharTraitsA::SafeStringLength(d));
        //    this->Write(d, l * sizeof(wchar_t));
        //}
        inline void Write(const vislib::RawStorage& d) {
            size_t l = d.GetSize();
            this->Write(static_cast<uint64_t>(l));
            this->Write(d, l);
        }
        template<class T> inline RawStorageWriter& operator<<(const T& d) {
            this->Write(d);
            return *this;
        }

    private:

        /**
         * Enforces the size 'e' to be available in the RawStorage. Calculates
         * the size increase of the RawStorage as multiple of 'inc'.
         *
         * @param e The new size to be available.
         */
        void assertSize(size_t e);

        /** The raw data store */
        vislib::RawStorage &data;

        /** The end-of-data pointer */
        size_t end;

        /** The memory size increment value */
        size_t inc;

        /** The position pointer */
        size_t pos;

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RAWSTORAGEWRITER_H_INCLUDED */

