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

		/** This type is used for the file size. */
		typedef UINT64 FileSize;

        /** Possible values for the access mode. */
        enum AccessMode { READ_WRITE = 0, READ_ONLY, WRITE_ONLY };

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
		static bool Delete(const char *filename);

        /**
         * Answer whether a file with the specified name exists.
         *
         * @param filename Path to the file to be tested.
         *
         * @return true, if the specified file exists, false otherwise.
         */
        static bool Exists(const char *filename);

		/** Ctor. */
		File(void);

		File(const File& rhs);

		/**
		 * Dtor. If the file is still open, it is closed.
		 */
		~File(void);

		/** Close the file, if open. */
		void Close(void);

		/**
		 * Forces all buffered data to be written. 
		 *
		 * @throws IOException
		 */
		void Flush(void);

		/**
		 * Answer the size of the file in bytes.
		 *
		 * @return The size of the file in bytes.
		 *
		 * @throws IOException If the file size cannot be retrieve, e. g. 
		 *                     because the file has not been opened.
		 */
		FileSize GetSize(void);

		/**
		 * Answer whether this file is open.
		 *
		 * @return true, if the file is open, false otherwise.
		 */
		bool IsOpen(void) const;

        //inline bool IsEOF(void) const {
        //    
        //}

		bool Open(const char *filename, const DWORD flags);

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
		FileSize Read(void *outBuf, const FileSize bufSize);

		/**
		 * Move the file pointer.
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
		FileSize Seek(const FileOffset offset, const SeekStartPoint from);

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
		 * Write 'bufSize' bytes from 'buf' to the file.
		 *
		 * @param buf     Pointer to the data to be written.
		 * @param bufSize The number of bytes to be written.
		 *
		 * @return The number of bytes acutally written.
		 *
	     * @throws IOException
		 */
		FileSize Write(const void *buf, const FileSize bufSize);

	private:

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
