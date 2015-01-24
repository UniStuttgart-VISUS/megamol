/*
 * RawStorageSerialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RawStorageSerialiser.h"

#include "vislib/assert.h"
#include "vislib/IllegalStateException.h"


/*
 * vislib::RawStorageSerialiser::RawStorageSerialiser
 */
vislib::RawStorageSerialiser::RawStorageSerialiser(RawStorage *storage, 
        unsigned int offset) 
        : Serialiser(SERIALISER_REQUIRES_ORDER), nakedStorage(NULL), 
        nakedStorageSize(0), storage(storage), offset(offset) {
    // Intentionally empty
}


/*
 * vislib::RawStorageSerialiser::RawStorageSerialiser
 */
vislib::RawStorageSerialiser::RawStorageSerialiser(const BYTE *storage, 
        const SIZE_T storageSize, const unsigned int offset) 
        : Serialiser(SERIALISER_REQUIRES_ORDER), nakedStorage(storage),
        nakedStorageSize(storageSize), storage(NULL), offset(offset) {
    // Intentionally empty
    ASSERT(storage != NULL);
}


/*
 * vislib::RawStorageSerialiser::RawStorageSerialiser
 */
vislib::RawStorageSerialiser::RawStorageSerialiser(
        const RawStorageSerialiser& src) 
        : Serialiser(src), nakedStorage(NULL), nakedStorageSize(0), 
        storage(NULL), offset(0) {
    *this = src;
}


/*
 * vislib::RawStorageSerialiser::~RawStorageSerialiser
 */
vislib::RawStorageSerialiser::~RawStorageSerialiser(void) {
    this->nakedStorage = NULL;  // DO NOT DELETE!
    this->nakedStorageSize = 0;
    this->storage = NULL;       // DO NOT DELETE!
    this->offset = 0;
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(bool& outValue, 
        const char *name) {
    unsigned char c;
    this->restore(&c, 1);
    outValue = (c != 0);
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(bool& outValue, 
        const wchar_t *name) {
    unsigned char c;
    this->restore(&c, 1);
    outValue = (c != 0);
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(wchar_t& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(wchar_t& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT8& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(INT8));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT8& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(INT8));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT8& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(UINT8));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT8& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(UINT8));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT16& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(INT16));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT16& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(INT16));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT16& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(UINT16));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT16& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(UINT16));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT32& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(INT32));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT32& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(INT32));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT32& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(UINT32));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT32& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(UINT32));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT64& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(INT64));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(INT64& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(INT64));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT64& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(UINT64));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(UINT64& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(UINT64));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(float& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(float));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(float& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(float));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(double& outValue, 
        const char *name) {
    this->restore(&outValue, sizeof(double));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(double& outValue, 
        const wchar_t *name) {
    this->restore(&outValue, sizeof(double));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(vislib::StringA& outValue,
        const char *name) {
    unsigned int len;
    this->restore(&len, sizeof(unsigned int));
    this->restore(outValue.AllocateBuffer(len), len * sizeof(char));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(vislib::StringA& outValue,
        const wchar_t *name) {
    unsigned int len;
    this->restore(&len, sizeof(unsigned int));
    this->restore(outValue.AllocateBuffer(len), len * sizeof(char));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(vislib::StringW& outValue,
        const char *name) {
    unsigned int len;
    this->restore(&len, sizeof(unsigned int));
    this->restore(outValue.AllocateBuffer(len), len * sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Deserialise
 */
void vislib::RawStorageSerialiser::Deserialise(vislib::StringW& outValue,
        const wchar_t *name) {
    unsigned int len;
    this->restore(&len, sizeof(unsigned int));
    this->restore(outValue.AllocateBuffer(len), len * sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const bool value, 
        const char *name) {
    unsigned char c = value ? 1 : 0;
    this->store(&c, 1);
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const bool value, 
        const wchar_t *name) {
    unsigned char c = value ? 1 : 0;
    this->store(&c, 1);
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const wchar_t value, 
        const char *name) {
    this->store(&value, sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const wchar_t value, 
        const wchar_t *name) {
    this->store(&value, sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT8 value, 
        const char *name) {
    this->store(&value, sizeof(INT8));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT8 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(INT8));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT8 value, 
        const char *name) {
    this->store(&value, sizeof(UINT8));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT8 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(UINT8));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT16 value, 
        const char *name) {
    this->store(&value, sizeof(INT16));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT16 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(INT16));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT16 value, 
        const char *name) {
    this->store(&value, sizeof(UINT16));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT16 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(UINT16));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT32 value, 
        const char *name) {
    this->store(&value, sizeof(INT32));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT32 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(INT32));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT32 value, 
        const char *name) {
    this->store(&value, sizeof(UINT32));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT32 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(UINT32));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT64 value, 
        const char *name) {
    this->store(&value, sizeof(INT64));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const INT64 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(INT64));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT64 value, 
        const char *name) {
    this->store(&value, sizeof(UINT64));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const UINT64 value, 
        const wchar_t *name) {
    this->store(&value, sizeof(UINT64));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const float value, 
        const char *name) {
    this->store(&value, sizeof(float));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const float value, 
        const wchar_t *name) {
    this->store(&value, sizeof(float));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const double value, 
        const char *name) {
    this->store(&value, sizeof(double));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const double value, 
        const wchar_t *name) {
    this->store(&value, sizeof(double));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const vislib::StringA& value, 
        const char *name) {
    unsigned int len = value.Length();
    this->store(&len, sizeof(unsigned int));
    this->store(value.PeekBuffer(), len * sizeof(char));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const vislib::StringA& value, 
        const wchar_t *name) {
    unsigned int len = value.Length();
    this->store(&len, sizeof(unsigned int));
    this->store(value.PeekBuffer(), len * sizeof(char));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const vislib::StringW& value, 
        const char *name) {
    unsigned int len = value.Length();
    this->store(&len, sizeof(unsigned int));
    this->store(value.PeekBuffer(), len * sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::Serialise
 */
void vislib::RawStorageSerialiser::Serialise(const vislib::StringW& value, 
        const wchar_t *name) {
    unsigned int len = value.Length();
    this->store(&len, sizeof(unsigned int));
    this->store(value.PeekBuffer(), len * sizeof(wchar_t));
}


/*
 * vislib::RawStorageSerialiser::SetOffset
 */
void vislib::RawStorageSerialiser::SetOffset(unsigned int offset) {
    this->offset = offset;
}


/*
 * vislib::RawStorageSerialiser::SetStorage
 */
void vislib::RawStorageSerialiser::SetStorage(vislib::RawStorage *storage) {
    this->storage = storage;
}


/*
 * vislib::RawStorageSerialiser::operator=
 */
vislib::RawStorageSerialiser& vislib::RawStorageSerialiser::operator =(
        const vislib::RawStorageSerialiser& rhs) {
    this->nakedStorage = rhs.nakedStorage;
    this->nakedStorageSize = rhs.nakedStorageSize;
    this->storage = rhs.storage;
    this->offset = rhs.offset;
    return *this;
}


/*
 * vislib::RawStorageSerialiser::store
 */
void vislib::RawStorageSerialiser::store(const void *data, unsigned int size) {
    if (this->storage == NULL) {
        throw vislib::IllegalStateException("No RawStorage object set",
            __FILE__, __LINE__);
    }
    this->storage->AssertSize(this->offset + size, true);
    memcpy(this->storage->As<char>() + this->offset, data, size);
    this->offset += size;
}


/*
 * vislib::RawStorageSerialiser::restore
 */
void vislib::RawStorageSerialiser::restore(void *data, unsigned int size) {
    /* Consolidate pointers. */
    if (this->storage != NULL) {
        this->nakedStorage = this->storage->As<BYTE>();
        this->nakedStorageSize = this->storage->GetSize();
    }

    /* Sanity checks. */
    if (this->nakedStorage == NULL) {
        throw vislib::IllegalStateException("Either a RawStorage object or a "
            "naked data pointer must be provided for deserialisation.",
            __FILE__, __LINE__);
    }
    if (this->nakedStorageSize < this->offset + size) {
        throw vislib::Exception("Not enough data in storage object to "
            "deserialise", __FILE__, __LINE__);
    }

    ::memcpy(data, this->nakedStorage + this->offset, size);
    this->offset += size;
}
