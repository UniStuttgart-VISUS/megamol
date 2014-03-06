/*
 * RawStorageSerialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RAWSTORAGESERIALISER_H_INCLUDED
#define VISLIB_RAWSTORAGESERIALISER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/RawStorage.h"
#include "vislib/Serialiser.h"


namespace vislib {


    /**
     * Implementation of serialiser operating on a raw storage. This 
     * serialiser uses the order of the elements to serialise or deserialise
     * 'Serialisable' objects.
     *
     * The serialiser is also able to deserialise from a naked data pointer.
     * However, it is not possible to serialise to a naked pointer.
     */
    class RawStorageSerialiser : public Serialiser {
    public:

        /** 
         * Ctor.
         *
         * @param storage Pointer to the RawStorage object to be used. The 
         *                serialiser does not become owner of the memory of the
         *                RawStorage object. The caller must ensure that the
         *                object remains valid as long as it is used by this
         *                serialiser or a copy of it.
         * @param offset The starting offset inside the RawStorage.
         */
        RawStorageSerialiser(RawStorage *storage = NULL, 
            unsigned int offset = 0);

        /**
         * Construct a serialiser that uses a naked data pointer. Such a 
         * serialiser is only good for deserialisation and cannot serialise.
         * Use a serialiser with RawStorage object for serialisation.
         *
         * @param storage     A pointer to 'storageSize' bytes of data that 
         *                    should be deserialised. The serialiser does not
         *                    take ownership of the object. The caller must 
         *                    ensure that the data remain valid as long as they
         *                    are used by this serialiser or a copy of it.
         * @param storageSize The number of bytes designated by 'storage'.
         */
        RawStorageSerialiser(const uint8_t *storage, const size_t storageSize,
            const unsigned int offset = 0);

        /**
         * Copy Ctor.
         *
         * @param src The source object to be cloned from.
         */
        RawStorageSerialiser(const RawStorageSerialiser& src);

        /** Dtor. */
        ~RawStorageSerialiser(void);

        virtual void Deserialise(bool& outValue, 
            const char *name = NULL);

        virtual void Deserialise(bool& outValue, 
            const wchar_t *name);

        virtual void Deserialise(wchar_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(wchar_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int8_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int8_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint8_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint8_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int16_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int16_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint16_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint16_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int32_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int32_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint32_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint32_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int64_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int64_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint64_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint64_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(float& outValue, 
            const char *name = NULL);

        virtual void Deserialise(float& outValue, 
            const wchar_t *name);

        virtual void Deserialise(double& outValue, 
            const char *name = NULL);

        virtual void Deserialise(double& outValue, 
            const wchar_t *name);

        virtual void Deserialise(StringA& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringA& outValue, 
            const wchar_t *name);

        virtual void Deserialise(StringW& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringW& outValue, 
            const wchar_t *name);

        /**
         * Answer the offset inside the rawstorage.
         *
         * @return The offset inside the rawstorage.
         */
        inline unsigned int Offset(void) const {
            return this->offset;
        }

        /**
         * Answer the RawStorage object pointer to be used.
         *
         * @return The RawStorage object pointer to be used.
         */
        RawStorage* Storage(void) const {
            return this->storage;
        }

        virtual void Serialise(const bool value, 
            const char *name = NULL);

        virtual void Serialise(const bool value, 
            const wchar_t *name);

        virtual void Serialise(const wchar_t value,
            const char *name = NULL);

        virtual void Serialise(const wchar_t value,
            const wchar_t *name);

        virtual void Serialise(const int8_t value,
            const char *name = NULL);

        virtual void Serialise(const int8_t value,
            const wchar_t *name);

        virtual void Serialise(const uint8_t value,
            const char *name = NULL);

        virtual void Serialise(const uint8_t value,
            const wchar_t *name);

        virtual void Serialise(const int16_t value,
            const char *name = NULL);

        virtual void Serialise(const int16_t value,
            const wchar_t *name);

        virtual void Serialise(const uint16_t value,
            const char *name = NULL);

        virtual void Serialise(const uint16_t value,
            const wchar_t *name);

        virtual void Serialise(const int32_t value,
            const char *name = NULL);

        virtual void Serialise(const int32_t value,
            const wchar_t *name);

        virtual void Serialise(const uint32_t value,
            const char *name = NULL);

        virtual void Serialise(const uint32_t value,
            const wchar_t *name);

        virtual void Serialise(const int64_t value,
            const char *name = NULL);

        virtual void Serialise(const int64_t value,
            const wchar_t *name);

        virtual void Serialise(const uint64_t value,
            const char *name = NULL);

        virtual void Serialise(const uint64_t value,
            const wchar_t *name);

        virtual void Serialise(const float value,
            const char *name = NULL);

        virtual void Serialise(const float value,
            const wchar_t *name);

        virtual void Serialise(const double value,
            const char *name = NULL);

        virtual void Serialise(const double value,
            const wchar_t *name);

        virtual void Serialise(const StringA& value,
            const char *name = NULL);

        virtual void Serialise(const StringA& value,
            const wchar_t *name);

        virtual void Serialise(const StringW& value,
            const char *name = NULL);

        virtual void Serialise(const StringW& value,
            const wchar_t *name);

        /**
         * Sets the offset inside the RawStorage to be used.
         *
         * @param offset The starting offset inside the RawStorage.
         */
        void SetOffset(unsigned int offset);

        /**
         * Sets the RawStorage to be used.
         *
         * @param storage Pointer to the RawStorage object to be used. The 
         *                serialiser does not become owner of the memory of the
         *                RawStorage object. The caller must ensure that the
         *                object remains valid as long as it is used by this
         *                serialiser.
         */
        void SetStorage(RawStorage *storage);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to 'this'.
         */
        RawStorageSerialiser& operator=(const RawStorageSerialiser& rhs);

    private:

        /**
         * Stores 'size' byte from 'data' to the current offset inside the
         * RawStorage and moves offset accordingly.
         *
         * @param data Pointer to the data to be stored.
         * @param size The number of bytes to be stored.
         */
        void store(const void *data, unsigned int size);

        /**
         * Restores 'size' byte from the current offset inside the RawStorage 
         * to 'data' and moves offset accordingly.
         *
         * @param data Pointer to the data to be restored.
         * @param size The number of bytes to be restored.
         */
        void restore(void *data, unsigned int size);

        /** Naked pointer to data to deserialise from. */
        const uint8_t *nakedStorage;

        /** Number of bytes designated by 'nakedStorage'. */
        size_t nakedStorageSize;

        /** Pointer to the raw storage object */
        RawStorage *storage;

        /** Offset inside the raw storage object */
        unsigned int offset;

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RAWSTORAGESERIALISER_H_INCLUDED */
