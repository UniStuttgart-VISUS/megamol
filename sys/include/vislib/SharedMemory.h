/*
 * SharedMemory.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHAREDMEMORY_H_INCLUDED
#define VISLIB_SHAREDMEMORY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <sys/types.h>
#endif /* _WIN32 */


#include "vislib/File.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * This class implements a shared memory segment that can be used for 
     * inter-process communication. This class handles both, the allocation
     * of the memory and its mapping.
     */
    class SharedMemory {

    public:

        /** The desired access to the shared memory segment. */
        enum AccessMode { 
            READ_WRITE = 1, 
            READ_ONLY
            //WRITE_ONLY 
        };

        /** The possible creation modes. */
        enum CreationMode { 
            CREATE_ONLY = 0,    // Fails, if the segment already exists.
            OPEN_ONLY,          // Fails, if the segment does not exist.
            OPEN_CREATE         // Opens existing or creates new file as needed.
        };

        /** This type is used for offsets when seeking in shared memory. */
        typedef File::FileOffset FileOffset;

        /** This type is used for size information of shared memory segments. */
        typedef File::FileSize FileSize;

        /** Ctor. */
        SharedMemory(void);

        /**
         * Dtor. 
         *
         * Destroying the object implies closing the segment.
         */
        ~SharedMemory(void);

        /**
         * Answer a typed pointer to the segment mapping.
         *
         * @return Pointer to the segment.
         */
        template<class T> inline T *As(void) {
            return static_cast<T *>(this->mapping);
        }

        /**
         * Answer a typed pointer to the segment mapping.
         *
         * @return Pointer to the segment.
         */
        template<class T> inline const T *As(void) const {
            return static_cast<T *>(this->mapping);
        }

        /**
         * Close the shared memory segment, if open. 
         *
         * If the object does not designate an open segment, the method does 
         * nothing.
         *
         * @throws SystemException If an existing shared memory segment could 
         *                         not be destroyed.
         */
        void Close(void);

        /**
         * Answer the shared memory segment is open and mapped.
         *
         * @return true in case the segment is open, false otherwise.
         */
        bool IsOpen(void) const;

        /**
         * Creates a new shared memory segment or opens an existing one.
         *
         * @param name         Name of the shared memory segment. This name 
         *                     should follow the kernel namespace rules of 
         *                     Windows. For Linux, a unique key is created
         *                     from the name.
         * @param accessMode   The desired access to the shared memory.
         * @param creationMode Specifies whether to create or to open a segment.
         * @param size         Size of the shared memory segment in bytes.
         *
         * @throws SystemException If the shared memory segment could not be
         *                         created or not be mapped.
         */
        void Open(const char *name, const AccessMode accessMode, 
            const CreationMode creationMode, const FileSize size);

        /**
         * Creates a new shared memory segment or opens an existing one.
         *
         * @param name         Name of the shared memory segment. This name 
         *                     should follow the kernel namespace rules of 
         *                     Windows. For Linux, a unique key is created
         *                     from the name.
         * @param accessMode   The desired access to the shared memory.
         * @param creationMode Specifies whether to create or to open a segment.
         * @param size         Size of the shared memory segment in bytes.
         *
         * @throws SystemException If the shared memory segment could not be
         *                         created or not be mapped.
         */
        void Open(const wchar_t *name, const AccessMode accessMode, 
            const CreationMode creationMode, const FileSize size);

        /**
         * Provides access to the mapped segment.
         *
         * @return Pointer to the segment.
         */
        inline operator void *(void) {
            return this->mapping;
        }

        /**
         * Provides access to the mapped segment.
         *
         * @return Pointer to the segment.
         */
        inline operator const void *(void) const {
            return this->mapping;
        }

    private:

#ifndef _WIN32
        /** The default permissions assigned to the shared memory segment. */
        static const mode_t DFT_MODE;
#endif /* !_WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        SharedMemory(const SharedMemory& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        SharedMemory& operator =(const SharedMemory& rhs);

#ifdef _WIN32
        /** Handle to the shared memory segment. */
        HANDLE hSharedMem;

#else /* _WIN32 */
        /** File descriptor of the shared memory segment. */
        int hSharedMem;

        /** Linux requires to preserve the name for unlinking the memory. */
        StringA name;

        /** Linux requires to preserve the mapping size for unmapping. */
        size_t size;
#endif /* _WIN32 */

        /** Pointer to the local mapping of the segment. */
        void *mapping;
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHAREDMEMORY_H_INCLUDED */
