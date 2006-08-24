/*
 * File.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FILE_H_INCLUDED
#define VISLIB_FILE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/types.h"
#include "vislib/tchar.h"


namespace vislib {
namespace sys {

    /**
     * Instances of this class repsesent a file. The class provides unbuffered
     * read and write access to the file. The implementation uses 64 bit 
     * operations on all platforms to allow access to very large files.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class File {

    public:
        
        /** This type is used for offsets when seeking in files. */
        typedef INT64 FileOffset;

        /** This type is used for size information of files. */
        typedef UINT64 FileSize;

        /** Possible values for the access mode. */
        enum AccessMode { READ_WRITE = 1, READ_ONLY, WRITE_ONLY };

        /** Possible values for the share mode. */
        enum ShareMode { SHARE_READ = 1, SHARE_WRITE = 2, SHARE_READWRITE = 3};

        /** Possible values for the CreationMode. */
        enum CreationMode { 
            CREATE_ONLY = 0,    // Fails, if file already exists.
            CREATE_OVERWRITE,   // Overwrites existing files.
            OPEN_ONLY,          // Fails, if file does not exist.
            OPEN_CREATE         // Opens existing or creates new file as needed.
        };

        /** Possible starting points for seek operations. */
        enum SeekStartPoint { 
#ifdef _WIN32
            BEGIN = FILE_BEGIN, 
            CURRENT = FILE_CURRENT, 
            END = FILE_END 
#else /* _WIN32 */
            BEGIN = SEEK_SET,
            CURRENT = SEEK_CUR,
            END = SEEK_END
#endif /* _WIN32 */
        };

        /**
         * Delete the file with the specified name.
         *
         * @param filename The name of the file to be deleted.
         *
         * @return
         */
        // TODO: Consider throwing exception with GetLastError(). Would allow 
        // user to know error reason without call to GetLastError().
        static bool Delete(const TCHAR *filename);

        /**
         * Answer whether a file with the specified name exists.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists, false otherwise.
         */
        static bool Exists(const TCHAR *filename);

        /**
         * Rename the file 'oldName' to 'newName'.
         *
         * @param oldName The name of the file to be renamed.
         * @param newName The new name of the file.
         *
         * @return true, if the file was found and renamed, false otherwise.
         */
        // TODO: Consider throwing exception with GetLastError(). Would allow 
        // user to know error reason without call to GetLastError().
        static bool Rename(const TCHAR *oldName, const TCHAR *newName);

        /** Ctor. */
        File(void);

        /**
         * Dtor. If the file is still open, it is closed.
         */
        virtual ~File(void);

        /** Close the file, if open. */
        virtual void Close(void);

        /**
         * Forces all buffered data to be written. 
         *
         * @throws IOException
         */
        virtual void Flush(void);

        /**
         * Answer the size of the file in bytes.
         *
         * @return The size of the file in bytes.
         *
         * @throws IOException If the file size cannot be retrieve, e. g. 
         *                     because the file has not been opened.
         */
        virtual FileSize GetSize(void) const;

        /**
         * Answer whether the file pointer is at the end of the file.
         *
         * @return true, if the eof flag is set, false otherwise.
         *
         * @throws IOException If the file is not open or the file pointer is at an
         *                     invalid position at the moment.
         */
        inline bool IsEOF(void) const {
            return (this->Tell() == this->GetSize());
        }

        /**
         * Answer whether this file is open.
         *
         * @return true, if the file is open, false otherwise.
         */
        bool IsOpen(void) const;

        /**
         * Opens a file.
         *
         * If this object already holds an open file, this file is closed (like 
         * calling Close) and the new file is opened.
         *
         * @param filename	   Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *					   (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return true, if the file has been successfully opened, false otherwise.
         *
         * @throws IllegalParamException
         */
        virtual bool Open(const TCHAR *filename, const AccessMode accessMode, 
            const ShareMode shareMode, const CreationMode creationMode);

        /**
         * Read at most 'bufSize' bytes from the file into 'outBuf'.
         *
         * @param outBuf  The buffer to receive the data.
         * @param bufSize The size of 'outBuf' in bytes.
         *
         * @return The number of bytes acutally read.
         *
         * @throws IOException
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
         *
         * @throws IOException If the file pointer could not be moved, e. g. 
         *                     because the file was not open or the offset was
         *                     invalid.
         */
        virtual FileSize Seek(const FileOffset offset, const SeekStartPoint from);

        /**
         * Move the file pointer to the begin of the file.
         *
         * @return The new offset of the file pointer from the beginning of the
         *         file. This should be zero ...
         *
         * @throws IOException If the file pointer could not be moved.
         */
        inline FileSize SeekToBegin(void) {
            return this->Seek(0, BEGIN);
        }

        /**
         * Move the file pointer to the end of the file.
         *
         * @return The new offset of the file pointer from the beginning of the
         *         file. This should be the size of the file.
         *
         * @throws IOException If the file pointer could not be moved.
         */
        inline FileSize SeekToEnd(void) {
            return this->Seek(0, END);
        }

        /**
         * Returns the position of the current file pointer
         *
         * @return Position of the file pointer in bytes from the beginning
         *         of the file.
         *
         * @throws IOException
         */
        virtual FileSize Tell(void) const;

        /**
         * Write 'bufSize' bytes from 'buf' to the file.
         *
         * @param buf     Pointer to the data to be written.
         * @param bufSize The number of bytes to be written.
         *
         * @return The number of bytes acutally written.
         *
         * @throws IOException
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
        File(const File& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        File& operator =(const File& rhs);

        /** The file handle. */
#ifdef _WIN32
        HANDLE handle;
#else /* _WIN32 */
        int handle;
#endif /* _WIN32 */
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_FILE_H_INCLUDED */
