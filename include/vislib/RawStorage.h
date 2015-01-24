/*
 * RawStorage.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RAWSTORAGE_H_INCLUDED
#define VISLIB_RAWSTORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {


    /**
     * This class wraps a block of dynamic memory that can also be growed
     * dynamically.
     */
    class RawStorage {

    public:

        /** 
         * Create a new raw storage block with the specified size.
         *
         * @param size The initial size in bytes.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        RawStorage(const SIZE_T size = 0);

        /**
         * Clone 'rhs'. This will deep copy the whole dynamic memory.
         *
         * @param rhs The object to be cloned.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        RawStorage(const RawStorage& rhs);

        /** Dtor. */
        ~RawStorage(void);

        /**
         * Append 'cntData' bytes of data beginning at 'data' to the end of the
         * raw storage. The object ensures that the specified amount of data can
         * be appended by enlarging the raw storage.
         *
         * The existing this->GetSize() bytes in the raw storage will not be
         * touched.
         *
         * The caller remains owner of the memory designated by 'data'. The 
         * object will create a deep copy in this method.
         *
         * @param data    The data to be appended. It is safe to pass a NULL 
         *                pointer for 'data'. In this case, nothing will be
         *                copied, but the size is adjusted like calling
         *                AssertSize(this->GetSize() + cntData, true).
         * @param cntData The number of bytes starting at 'data' to be added.
         *
         * @return A pointer to the copy of 'data' in the raw storage. The 
         *         object remains owner of this data.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        void *Append(const void *data, const SIZE_T cntData);

        /**
         * Answer the raw memory block as pointer to T. The object remains
         * owner of the memory designated by the pointer returned.
         *
         * @return The raw memory block.
         */
        template<class T> inline T *As(void) {
            return reinterpret_cast<T *>(this->data);
        }

        /**
         * Answer the raw memory block as pointer to T. The object remains
         * owner of the memory designated by the pointer returned.
         *
         * @return The raw memory block.
         */
        template<class T> inline const T *As(void) const {
            return reinterpret_cast<const T*>(this->data);
        }

        /**
         * Answer the raw memory block at an offset of 'offset' bytes as 
         * pointer to T. The object remains owner of the memory designated 
         * by the pointer returned.
         *
         * Note that the method does not perform any range checks!
         *
         * @return A pointer to the begin of the memory block + 'offset' bytes.
         */
        template<class T> inline T *AsAt(const SIZE_T offset) {
            return reinterpret_cast<T *>(static_cast<BYTE *>(this->data) 
                + offset);
        }

        /**
         * Answer the raw memory block at an offset of 'offset' bytes as 
         * pointer to T. The object remains owner of the memory designated 
         * by the pointer returned.
         *
         * Note that the method does not perform any range checks!
         *
         * @return A pointer to the begin of the memory block + 'offset' bytes.
         */
        template<class T> inline const T *AsAt(const SIZE_T offset) const {
            return reinterpret_cast<T *>(static_cast<BYTE *>(this->data) 
                + offset);
        }

        /**
         * Ensures that the dynamic memory consists of at least 'size' bytes.
         * If the memory block already has the requested size or is even larger,
         * nothing is done. If 'keepContent' is true, the content of the 
         * memory will be preserved when reallocating. Otherwise, it might be
         * deleted.
         *
         * @param size        The new size in bytes.
         * @param keepContent Set this true, for reallocating and keeping the
         *                    content.
         *
         * @return true, if the memory was reallocated, false, if the memory 
         *         block was already large enough.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        bool AssertSize(const SIZE_T size, const bool keepContent = false);

        /**
         * Answer a pointer to the data at 'offset' byte distance from the 
         * beginning of the raw storage.
         *
         * Note that the method does not perform any range checks!
         *
         * @return A pointer to the begin of the memory block + 'offset' bytes.
         */
        inline void *At(const SIZE_T offset) {
            return (static_cast<BYTE *>(this->data) + offset);
        }

        /**
         * Answer a pointer to the data at 'offset' byte distance from the 
         * beginning of the raw storage.
         *
         * Note that the method does not perform any range checks!
         *
         * @return A pointer to the begin of the memory block + 'offset' bytes.
         */
        inline const void *At(const SIZE_T offset) const {
            return (static_cast<BYTE *>(this->data) + offset);
        }

        /**
         * Enforces the dynamic memory to contain exactly 'size' bytes. If
         * 'keepContent' is true, the memory will be reallocated so that
         * the current content remains. Otherwise, the content will be destroyed
         * and newly allocated.
         *
         * @param size        The new size in bytes.
         * @param keepContent Set this true, for reallocating and keeping the
         *                    content.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        void EnforceSize(const SIZE_T size, const bool keepContent = false);

        /**
         * Answer the size of the memory block.
         *
         * @return The size of the memory block in bytes.
         */
        inline SIZE_T GetSize(void) const {
            return this->size;
        }

        /**
         * Answer whether the memory block is empty, i. e. has no memory 
         * allocated.
         *
         * @return true, if no memory has been allocated, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->size == 0);
        }

        /**
         * Answer whether the object has at least 'size' bytes.
         *
         * @param size The minimum size to be tested.
         *
         * @return true, if the memory the memory block is 'size' bytes or larger,
		 *         false otherwise.
         */
		inline bool TestSize(const SIZE_T size) {
			return (this->size >= size);
		}

        /**
         * Sets the whole allocated memory to zero.
         */
        void ZeroAll(void);

        /**
         * Assignment operation. This will deep copy the whole dynamic memory.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws std::bad_alloc If the requested memory could not be allocated.
         */
        RawStorage& operator =(const RawStorage& rhs);

        /**
         * Test for equality. The content of the memory will be compared using
         * memcmp.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const RawStorage& rhs) const;

        /**
         * Test for inequality. The content of the memory will be compared using
         * memcmp.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const RawStorage& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Answer a pointer to the actual memory. The object remains
         * owner of the memory designated by the pointer returned.
         *
         * @return A pointer to the memory.
         */
        inline operator void *(void) {
            return this->data;
        }

        /**
         * Answer a pointer to the actual memory. The object remains
         * owner of the memory designated by the pointer returned.
         *
         * @return A pointer to the memory.
         */
        inline operator const void *(void) const {
            return this->data;
        }

    private:

        /** Pointer to dynamic memory. */
        void *data;

        /** The size of the memory block 'data' in bytes. */
        SIZE_T size;

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RAWSTORAGE_H_INCLUDED */
