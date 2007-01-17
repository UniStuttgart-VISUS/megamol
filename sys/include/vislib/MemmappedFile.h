/*
 * MemmappedFile.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MEMMAPPED_FILE_H_INCLUDED
#define VISLIB_MEMMAPPED_FILE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/File.h"

namespace vislib {
namespace sys {

    /**
     * Instances of this class repsesent a file based on vislib::sys::File but
     * with buffered access for reading and writing.
     *
     * @author Guido Reina (guido.reina@vis.uni-stuttgart.de)
     */
    class MemmappedFile : public File {
    public:

        ///**
        // * Answers the default size used when creating new BufferedFile objects
        // *
        // * @return size in bytes used when creating new buffers
        // */
        //inline static File::FileSize GetDefaultBufferSize(void) {
        //    return BufferedFile::defaultBufferSize;
        //}

        ///**
        // * Sets the default size used when creating new BufferedFile objects
        // *
        // * @param newSize The new size in bytes used when creating new buffers
        // */
        //inline static void SetDefaultBufferSize(File::FileSize newSize) {
        //    BufferedFile::defaultBufferSize = newSize;
        //}

        /** Ctor. */
        MemmappedFile(void);

        /**
         * Dtor. If the file is still open, it is closed.
         */
        virtual ~MemmappedFile(void);

        /** Close the file, if open. */
        virtual void Close(void);

        /**
         * behaves like File::Flush, except that it flushes only dirty buffers.
         *
         * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
         *                    could not be flushed to disk.
         */
        virtual void Flush(void);

        /**
         * Returns the size of the current view in bytes.
         * The number of bytes of valid data in this buffer can differ.
         *
         * @return number of bytes in view
         */
        inline File::FileSize GetViewSize(void) const {
			return this->viewSize;
        }

        /**
         * behaves like File::GetSize
         */
        virtual File::FileSize GetSize(void) const;

        /**
         * behaves like File::Open except for WRITE_ONLY files. These are silently upgraded to READ_WRITE
		 * since there is no such thing as a memory-mapped WRITE_ONLY file.
         */
        virtual bool Open(const char *filename, const File::AccessMode accessMode, 
            const File::ShareMode shareMode, const File::CreationMode creationMode);

        /**
         * behaves like File::Open except for WRITE_ONLY files. These are silently upgraded to READ_WRITE
		 * since there is no such thing as a memory-mapped WRITE_ONLY file.
         */
        virtual bool Open(const wchar_t *filename, const File::AccessMode accessMode, 
            const File::ShareMode shareMode, const File::CreationMode creationMode);

        /**
         * behaves like File::Read
         * Performs an implicit flush if the buffer is not in read mode.	// BUG: DO WE NEED TO?
         * Ensures that the buffer is in read mode.							// BUG: I WANT TRUE READWRITE FILES
         */
        virtual File::FileSize Read(void *outBuf, const File::FileSize bufSize);

        /**
         * behaves like File::Seek
         * Performs an implicit flush if the buffer is in write mode.		// BUG: ONLY WHEN DIRTY
         */
        virtual File::FileSize Seek(const File::FileOffset offset, const File::SeekStartPoint from);

        /**
         * Sets the size of the current view.
         * Calling this method implictly flushes the buffer.
		 * The size is automatically rounded to the next smaller multiple of AllocationGranularity.
		 *
		 * @return the actually set size
         *
         * @param newSize The number of bytes to be used for the new view.
         *
         * @throws IOException if the flush cannot be performed
         */
         File::FileSize SetViewSize(File::FileSize newSize);

        /**
         * behaves like File::Tell
         */
        virtual File::FileSize Tell(void) const;

        /**
         * behaves like File::Write
         * Performs an implicit flush if the buffer is not in write mode.	// BUG: nah.
         * Ensures that the buffer is in write mode.						// BUG: nah. depends on access
         *
         * @throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
         *                    could not be flushed to disk.
         */
        virtual File::FileSize Write(const void *buf, const File::FileSize bufSize);

    private:

		inline void safeUnmap();

		inline void safeCloseMapping();

        ///** Possible values for the buffer mode */
        //enum BufferMode { VOID_BUFFER, READ_BUFFER, WRITE_BUFFER };

        /**
         * Forbidden copy-ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        MemmappedFile(const MemmappedFile& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        MemmappedFile& operator =(const MemmappedFile& rhs);

		/**
		 * I do not like to implement the open twice. So here goes the common code.
		 */
		bool commonOpen(const File::AccessMode accessMode);

        ///** the default buffer size when creating new buffers */
        //static File::FileSize defaultBufferSize;

        ///** the buffer for IO */
        //unsigned char *buffer;

        /** 
         * the starting position of the view in bytes from the beginning of 
         * the file 
         */
        File::FileSize viewStart;

        /** the size of the view in bytes */
        File::FileSize viewSize;

		/**
		 * pointer to the mapped data
		 */
		char *mappedData;

		/**
		 * mapping used to generate views
		 */
#ifdef _WIN32
		HANDLE mapping;
#else /* _WIN32 */
		// BUG: data type???
#endif /* _WIN32 */

		/**
		 * writing needs this parameter for each now mapping...
		 */
		DWORD protect;

		/**
		 * store the access mode for later reference (e.g. views)
		 */
		File::AccessMode access;

        ///** the mode of the current buffer */
        //BufferMode bufferMode;

        /** the absolute position inside the file */
        File::FileSize filePos;

		/**
		 * virtual eof position needed for emulating normal files since we can only generate multiples
		 * of AllocationGranularity
		 */
		File::FileSize endPos;

		/** unless I am sure of what happens with file sizes that do not match the granularity... */
		File::FileSize referenceSize;

		/** does a writable view have to be written back to disk */
		bool viewDirty;

        ///** the number of bytes of the buffer which hold valid informations */
        //File::FileSize validBufferSize;

    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_MEMMAPPED_FILE_H_INCLUDED */
