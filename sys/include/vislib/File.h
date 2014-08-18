/*
 * File.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FILE_H_INCLUDED
#define VISLIB_FILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifndef _WIN32
#include <unistd.h>
#endif /* !_WIN32 */

#ifdef _MSC_VER
#pragma comment(lib, "shlwapi")
#endif /* _MSC_VER */


#include "vislib/String.h"
#include "vislib/types.h"


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
        enum ShareMode { 
            SHARE_EXCLUSIVE = 1, 
            SHARE_READ, 
            SHARE_WRITE, 
            SHARE_READWRITE
        };

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
         * Creates a temporary file. The file has 'AccessMode' 'READ_WRITE'.
         * Because the name of the file is highly OS configuration dependent
         * there is no way of opening the file a second time. Therefore the
         * 'ShareMode' is 'SHARE_EXCLUSIVE'. The content of the temporary file
         * will be held in memory as long as the cache size is sufficient. The
         * file on secondary storage will be deleted when the file is closed.
         * This will also happen if 'Open' is called on the returned object.
         *
         * @return A pointer to the 'File' object of the created temporary
         *         file. This object is placed on the heap and the caller must
         *         delete it when it is no longer needed (best practice is to
         *         assign the returned pointer to a 'SmartPtr' object). The
         *         return value is 'NULL' if there was an unexpected error.
         *
         * @throws SystemException in most error cases.
         */
        static File* CreateTempFile(void);

        /**
         * Creates a file name for a temporary file
         *
         * @param outFn the string to receive the file name created
         *
         * @return outFn
         */
        static vislib::StringA& CreateTempFileName(vislib::StringA& outFn);

        /**
         * Creates a file name for a temporary file
         *
         * @param outFn the string to receive the file name created
         *
         * @return outFn
         */
        static vislib::StringW& CreateTempFileName(vislib::StringW& outFn);

        /**
         * Creates a file name for a temporary file
         *
         * @return the file name created
         */
        static inline vislib::StringA CreateTempFileNameA(void) {
            vislib::StringA s;
            return CreateTempFileName(s);
        }

        /**
         * Creates a file name for a temporary file
         *
         * @return the file name created
         */
        static inline vislib::StringW CreateTempFileNameW(void) {
            vislib::StringW s;
            return CreateTempFileName(s);
        }

        /**
         * Delete the file with the specified name.
         *
         * @param filename The name of the file to be deleted.
         *
         * @return true in case of success, false otherwise. Use 
         *         ::GetLastError() to retrieve further information on 
         *         failure.
         */
        static bool Delete(const char *filename);

        /**
         * Delete the file with the specified name.
         *
         * @param filename The name of the file to be deleted.
         *
         * @return true in case of success, false otherwise. Use 
         *         ::GetLastError() to retrieve further information on 
         *         failure.
         */
        static bool Delete(const wchar_t *filename);

        /**
         * Answer whether a file with the specified name exists.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists, false otherwise.
         */
        static bool Exists(const char *filename);

        /**
         * Answer whether a file with the specified name exists.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists, false otherwise.
         */
        static bool Exists(const wchar_t *filename);

        /**
         * Answer the size of a file
         *
         * @param filename Path to the file
         *
         * @return The size of the file.
         *
         * @throws SystemException in most error cases.
         */
        static FileSize GetSize(const char *filename);

        /**
         * Answer the size of a file
         *
         * @param filename Path to the file
         *
         * @return The size of the file.
         *
         * @throws SystemException in most error cases.
         */
        static FileSize GetSize(const wchar_t *filename);

        /**
         * Answer whether a file with the specified name is a directory.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists and is a directory,
         *         false otherwise.
         */
        static bool IsDirectory(const char *filename);

        /**
         * Answer whether a file with the specified name is a directory.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists and is a directory,
         *         false otherwise.
         */
        static bool IsDirectory(const wchar_t *filename);

        /**
         * Answer whether a file with the specified name is a normal file (not 
         * a directory or any other special file system element).
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists and is a normal file,
         *         false otherwise.
         */
        static bool IsFile(const char *filename);

        /**
         * Answer whether a file with the specified name is a normal file (not 
         * a directory or any other special file system element).
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists and is a normal file,
         *         false otherwise.
         */
        static bool IsFile(const wchar_t *filename);

        /**
         * Rename the file 'oldName' to 'newName'.
         *
         * @param oldName The name of the file to be renamed.
         * @param newName The new name of the file.
         *
         * @return true in case of success, false otherwise. Use 
         *         ::GetLastError() to retrieve further information on 
         *         failure.
         */
        static bool Rename(const char *oldName, const char *newName);

        /**
         * Rename the file 'oldName' to 'newName'.
         *
         * @param oldName The name of the file to be renamed.
         * @param newName The new name of the file.
         *
         * @return true in case of success, false otherwise. Use 
         *         ::GetLastError() to retrieve further information on 
         *         failure.
         */
        static bool Rename(const wchar_t *oldName, const wchar_t *newName);

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
        virtual bool IsOpen(void) const;

        /**
         * Opens a file.
         *
         * If this object already holds an open file, this file is closed (like 
         * calling Close) and the new file is opened.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return true, if the file has been successfully opened, false 
         *         otherwise. In case of an error you can receive additional
         *         information using 'GetLastError'.
         *
         * @throws IllegalParamException
         */
        virtual bool Open(const char *filename, const AccessMode accessMode, 
            const ShareMode shareMode, const CreationMode creationMode);

        /**
         * Opens a file.
         *
         * If this object already holds an open file, this file is closed (like 
         * calling Close) and the new file is opened.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return true, if the file has been successfully opened, false 
         *         otherwise. In case of an error you can receive additional
         *         information using 'GetLastError'.
         *
         * @throws IllegalParamException
         */
        inline bool Open(const StringA& filename, const AccessMode accessMode, 
                const ShareMode shareMode, const CreationMode creationMode) {
            return this->Open(filename.PeekBuffer(), accessMode, shareMode, 
                creationMode);
        }

        /**
         * Opens a file.
         *
         * If this object already holds an open file, this file is closed (like 
         * calling Close) and the new file is opened.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return true, if the file has been successfully opened, false 
         *         otherwise. In case of an error you can receive additional
         *         information using 'GetLastError'.
         *
         * @throws IllegalParamException
         */
        virtual bool Open(const wchar_t *filename, const AccessMode accessMode, 
            const ShareMode shareMode, const CreationMode creationMode);

        /**
         * Opens a file.
         *
         * If this object already holds an open file, this file is closed (like 
         * calling Close) and the new file is opened.
         *
         * @param filename     Path to the file to be opened
         * @param accessMode   The access mode for the file to be opened
         * @param shareMode    The share mode
         *                     (Parameter is ignored on linux systems.)
         * @param creationMode Use your imagination on this one
         *
         * @return true, if the file has been successfully opened, false 
         *         otherwise. In case of an error you can receive additional
         *         information using 'GetLastError'.
         *
         * @throws IllegalParamException
         */
        inline bool Open(const StringW& filename, const AccessMode accessMode, 
                const ShareMode shareMode, const CreationMode creationMode) {
            return this->Open(filename.PeekBuffer(), accessMode, shareMode, 
                creationMode);
        }

        /**
         * Read at most 'bufSize' bytes from the file into 'outBuf'.
         *
         * @param outBuf  The buffer to receive the data.
         * @param bufSize The size of 'outBuf' in bytes.
         *
         * @return The number of bytes actually read.
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
        virtual FileSize Seek(const FileOffset offset, 
            const SeekStartPoint from = BEGIN);

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

    protected:

        /** The file handle. */
#ifdef _WIN32
        HANDLE handle;
#else /* _WIN32 */
        int handle;
#endif /* _WIN32 */
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FILE_H_INCLUDED */
