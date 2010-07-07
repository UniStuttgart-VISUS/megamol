/*
 * BufferedFile.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BUFFERED_FILE_H_INCLUDED
#define VISLIB_BUFFERED_FILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/File.h"


namespace vislib {
namespace sys {

    /**
     * Instances of this class repsesent a file based on vislib::sys::File but
     * with buffered access for reading and writing.
     *
     * @author Sebastian Grottel (sebastian.grottel@vis.uni-stuttgart.de)
     */
    class BufferedFile : public File {
    public:

        /**
         * Answers the default size used when creating new BufferedFile objects
         *
         * @return size in bytes used when creating new buffers
         */
        inline static File::FileSize GetDefaultBufferSize(void) {
            return BufferedFile::defaultBufferSize;
        }

        /**
         * Sets the default size used when creating new BufferedFile objects
         *
         * @param newSize The new size in bytes used when creating new buffers
         */
        inline static void SetDefaultBufferSize(File::FileSize newSize) {
            BufferedFile::defaultBufferSize = newSize;
        }

        /** Ctor. */
        BufferedFile(void);

        /**
         * Dtor. If the file is still open, it is closed.
         */
        virtual ~BufferedFile(void);

        /** Close the file, if open. */
        virtual void Close(void);

        /**
         * behaves like File::Flush
         *
         * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
         *                    could not be flushed to disk.
         */
        virtual void Flush(void);

        /**
         * Answer the size of the current buffer in bytes.
         * The number of bytes of valid data in this buffer can differ.
         *
         * @return number of bytes of current buffer
         */
        inline File::FileSize GetBufferSize(void) const {
            return this->bufferSize;
        }

        /**
         * behaves like File::GetSize
         */
        virtual File::FileSize GetSize(void) const;

        /**
         * behaves like File::Open
         */
        virtual bool Open(const char *filename,
            const File::AccessMode accessMode,
            const File::ShareMode shareMode,
            const File::CreationMode creationMode);

        /**
         * behaves like File::Open
         */
        virtual bool Open(const wchar_t *filename,
            const File::AccessMode accessMode,
            const File::ShareMode shareMode,
            const File::CreationMode creationMode);

        /**
         * behaves like File::Read
         * Performs an implicite flush if the buffer is not in read mode.
         * Ensures that the buffer is in read mode.
         */
        virtual File::FileSize Read(void *outBuf,
            const File::FileSize bufSize);

        /**
         * behaves like File::Seek
         * Performs an implicite flush if the buffer is in write mode.
         */
        virtual File::FileSize Seek(const File::FileOffset offset,
            const File::SeekStartPoint from = File::BEGIN);

        /**
         * Sets the size of the current buffer.
         * Calling this methode implictly flushes the buffer.
         *
         * @param newSize The number of bytes to be used for the new buffer.
         *
         * @throws IOException if the flush cannot be performed
         */
        void SetBufferSize(File::FileSize newSize);

        /**
         * behaves like File::Tell
         */
        virtual File::FileSize Tell(void) const;

        /**
         * behaves like File::Write
         * Performs an implicite flush if the buffer is not in write mode.
         * Ensures that the buffer is in write mode.
         *
         * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
         *                    could not be flushed to disk.
         */
        virtual File::FileSize Write(const void *buf,
            const File::FileSize bufSize);

    private:

        /** the default buffer size when creating new buffers */
        static File::FileSize& defaultBufferSize;

        /**
         * Forbidden copy-ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        BufferedFile(const BufferedFile& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        BufferedFile& operator =(const BufferedFile& rhs);

        /**
         * behaves like File::Flush
         *
         * @param fileFlush Flag whether or not to flush the file.
         *
         * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
         *                    could not be flushed to disk.
         */
        void flush(bool fileFlush);

        /** Resets the buffer. */
        void resetBuffer();

        /** the buffer for IO */
        unsigned char *buffer;

        /** the position inside the buffer */
        File::FileSize bufferOffset;

        /** the size of the buffer in bytes */
        File::FileSize bufferSize;

        /** 
         * the starting position of the buffer inside the file in bytes from 
         * the beginning of the file.
         */
        File::FileSize bufferStart;

        /** flag wether or not the buffer is dirty. */
        bool dirtyBuffer;

        /** the access mode the file has been opened with */
        File::AccessMode fileMode;

        /** the number of bytes of the buffer which hold valid informations */
        File::FileSize validBufferSize;

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BUFFERED_FILE_H_INCLUDED */
