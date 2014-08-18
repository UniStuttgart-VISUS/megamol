/*
 * MemoryFile.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MEMORYFILE_H_INCLUDED
#define VISLIB_MEMORYFILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/File.h"
#include "vislib/RawStorage.h"


namespace vislib {
namespace sys {


    /**
     * File accessor to main memory
     */
    class MemoryFile : public File {
    public:

        /** Ctor. */
        MemoryFile(void);

        /** Dtor. */
        virtual ~MemoryFile(void);

        /** Close the file, if open. */
        virtual void Close(void);

        /**
         * Forces all buffered data to be written.
         * This method has no effect.
         *
         * @throws IOException
         */
        virtual void Flush(void);

        /**
         * Answer the size of the file in bytes.
         *
         * @return The size of the file in bytes.
         */
        virtual FileSize GetSize(void) const;

        /**
         * Answer whether this file is open.
         *
         * @return true, if the file is open, false otherwise.
         */
        virtual bool IsOpen(void) const;

        /**
         * Opens a flat memory pointer as file. This object will not take
         * ownership of the memory 'buffer' points to. The caller is
         * responsible to keep the memory and the pointer valid as long as it
         * is used by this object.
         *
         * @param buffer The pointer to the memory. Must not be NULL
         * @param bufferlength The size of the memory 'buffer' points to in
         *                     bytes
         * @param accessMode The access mode for the memory file
         *
         * @return true on success, false on failure
         */
        virtual bool Open(void *buffer, SIZE_T bufferLength,
            AccessMode accessMode);

        /**
         * Opens a RawStorage as file. This object will not take ownership of
         * the RawStorage object. The caller is responsible to keep the object
         * alive and valid as long as it is used by this object. Resizing the
         * RawStorage object while it is used by this class may result in
         * undefined behaviour of any methods of this class.
         *
         * @param storage The RawStorage to be used as file
         * qparam accessMode The access mode for the memory file
         *
         * @return true on success, false on failure
         */
        virtual bool Open(RawStorage& storage, AccessMode accessMode);

        /**
         * Opens a file.
         * This method has no effect and will always return false.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return false
         */
        virtual bool Open(const char *filename, const AccessMode accessMode, 
            const ShareMode shareMode, const CreationMode creationMode);

        /**
         * Opens a file.
         * This method has no effect and will always return false.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return false
         */
        virtual bool Open(const wchar_t *filename, const AccessMode accessMode, 
            const ShareMode shareMode, const CreationMode creationMode);

        /** redeclare remaining Open overloads from file */
        using File::Open;

        /**
         * Read at most 'bufSize' bytes from the file into 'outBuf'.
         *
         * @param outBuf  The buffer to receive the data.
         * @param bufSize The size of 'outBuf' in bytes.
         *
         * @return The number of bytes actually read.
         */
        virtual FileSize Read(void *outBuf, const FileSize bufSize);

        /**
         * Move the file pointer.
         *
         * If the file pointer is seeked beyond the end of file, the behaviour is 
         * undefined for Read, Write, Tell and isEoF
         *
         * @param offset The offset in bytes.
         * @param from   The begin of the seek operation, which can be one of
         *               BEGIN, CURRENT, or END.
         *
         * @return The new offset in bytes of the file pointer from the begin of 
         *         the file.
         */
        virtual FileSize Seek(const FileOffset offset, 
            const SeekStartPoint from = BEGIN);

        /**
         * Returns the position of the current file pointer
         *
         * @return Position of the file pointer in bytes from the beginning
         *         of the file.
         */
        virtual FileSize Tell(void) const;

        /**
         * Write 'bufSize' bytes from 'buf' to the file.
         * If the data to be written does not fit into the remaining memory
         * the behaviour depends on the used memory object. When using a
         * rawstorage object the memory is extended to fit all data. When
         * using a void pointer only as much data is written as fits into
         * the allocated memory.
         *
         * @param buf     Pointer to the data to be written.
         * @param bufSize The number of bytes to be written.
         *
         * @return The number of bytes acutally written.
         */
        virtual FileSize Write(const void *buf, const FileSize bufSize);

    private:

        /**
         * Forbidden copy-ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        MemoryFile(const MemoryFile& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        MemoryFile& operator=(const MemoryFile& rhs);

        /** The memory file access mode */
        AccessMode accessMode;

        /** The flat buffer */
        unsigned char *buffer;

        /** The length of the flat buffer */
        FileSize bufferLen;

        /** The current position */
        FileSize pos;

        /** The raw storage object used */
        RawStorage *storage;

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MEMORYFILE_H_INCLUDED */

