/*
 * MemmappedFile.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MEMMAPPED_FILE_H_INCLUDED
#define VISLIB_MEMMAPPED_FILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/File.h"


#ifndef _WIN32
#define __USE_FILE_OFFSET64
#include <sys/mman.h>
#endif /* _WIN32 */

namespace vislib {
namespace sys {

    /**
     * Instances of this class repsesent a file based on vislib::sys::File but
     * with memory-mapped access for reading and writing. Memory-mapped I/O is very efficient and
     * recommended for reading. Read/write access is fully supported but should be
     * used with caution. Write-only access does not exist and is therefore silently upgraded to
     * read/write.
     *
     * @author Guido Reina (guido.reina@vis.uni-stuttgart.de)
     */
    class MemmappedFile : public File {
    public:

        /** Ctor. */
        MemmappedFile(void);

        /**
         * Dtor. If the file is still open, it is closed.
         */
        virtual ~MemmappedFile(void);

        /** Close the file, if open. Flush, if necessary. */
        virtual void Close(void);

        /**
         * behaves like File::Flush, except that it flushes only dirty buffers.
         *
         * @throws IOException with ERROR_WRITE_FAULT (EFBIG on linux) if a buffer in write mode
         *                    could not be flushed to disk. GetLastError() will provide details (Windows).
         * @throws IOException with details (linux).
         */
        virtual void Flush(void);

        /**
         * Returns the size of the current view in bytes.
         * The number of bytes of valid data in this buffer can differ.
         *
         * @return number of bytes in view
         */
        virtual inline File::FileSize GetViewSize(void) const {
            return this->viewSize;
        }

        /**
         * behaves mostly like File::GetSize
         *
         * @throws IllegalStateException if the file is not open
         */
        virtual File::FileSize GetSize(void) const;

        /**
         * behaves like File::Open except for WRITE_ONLY files. These are silently upgraded to READ_WRITE
         * since there is no such thing as a memory-mapped WRITE_ONLY file (also under linux even if no one
         * tells you that there is NO SUCH THING as a O_WRONLY memmapped file *hate*).
         *
         * @throws IOException
         */
        virtual bool Open(const char *filename, const File::AccessMode accessMode, 
            const File::ShareMode shareMode, const File::CreationMode creationMode);

        /**
         * behaves like File::Open except for WRITE_ONLY files. These are silently upgraded to READ_WRITE
         * since there is no such thing as a memory-mapped WRITE_ONLY file (also under linux even if no one
         * tells you that there is NO SUCH THING as a O_WRONLY memmapped file *hate*).
         *
         * @throws IOException
         */
        virtual bool Open(const wchar_t *filename, const File::AccessMode accessMode, 
            const File::ShareMode shareMode, const File::CreationMode creationMode);

        /**
         * behaves like File::Read
         * Performs an implicit flush if the view is dirty and changed.
         *
         * @throws IOException ERROR_WRITE_FAULT (EFBIG on linux) if flushing
         * @throws IOException ERROR_ACCESS_DENIED (windows) or EACCES (linux) if called on write-only files
         * @throws IOException on mapping failures. Use GetLastError().
         * @throws IllegalStateException if file is not open, or access mode unsuitable, for example
         */
        virtual File::FileSize Read(void *outBuf, const File::FileSize bufSize);

        /**
         * behaves like File::Seek
         * If the destination is beyond file extents, it is cropped.
         * No flush is performed since changing views already takes care of that.
         */
        virtual File::FileSize Seek(const File::FileOffset offset, 
            const File::SeekStartPoint from = File::BEGIN);

        /**
         * Sets the size of the current view.
         * Calling this method implictly unmaps the view (and flushes the buffer).
         * The size is automatically rounded to the next smaller, greater-than-zero multiple of AllocationGranularity.
         *
         * @return the actually set size
         *
         * @param newSize The number of bytes to be used for the new view.
         *
         * @throws see Flush()
         */
         File::FileSize SetViewSize(File::FileSize newSize);

        /**
         * behaves like File::Tell
         *
         * @throws IllegalStateException if the file is not open
         */
        virtual File::FileSize Tell(void) const;

        /**
         * behaves like File::Write
         * Performs an implicit flush if the view is dirty and changed.
         *
         * @throws IOException ERROR_WRITE_FAULT (EFBIG on linux) if flushing fails
         * @throws IOException ERROR_ACCESS_DENIED (EACCES on linux) if called on read-only files
         * @throws IOException on mapping failures. Use GetLastError().
         * @throws IllegalStateException if file is not open, or access mode unsuitable, for example
         */
        virtual File::FileSize Write(const void *buf, const File::FileSize bufSize);

    private:

        /**
         * Generate a valid view size in relation to a file pointer position
         *
         * @param pos the (unaligned) position
         *
         * @return the corrected view size
         */
        inline File::FileSize AdjustedViewSize(File::FileSize pos);

        /**
         * If file is in write mode, this calls for a Flush(). Then it unmaps the current view,
         * if there is any. mappedData will be NULL afterwards.
         *
         * @throws IOException if unmapping fails. Use GetLastError().
         */
        inline void SafeUnmapView();


#ifdef _WIN32
        /**
         * Creates a this->mapping for this->handle. This is only needed on windows.
         * Will update this->referenceSize  to fit the new size or if the handle generated is the clone of another handle to
         * an already open file.
         *
         * @param mappingsize the maximum file size to be mapped; this will be available for views.
         *
         * @throws IOException if creation fails.
         */
        inline void SafeCreateMapping(File::FileSize mappingsize);


        /**
         * Closes the mapping applied to handle. If views still exist, the mapping will persist
         * until these are unmapped as well.
         *
         * @throws IOException if closing fails. Use GetLastError().
         */
        inline void SafeCloseMapping();
#endif /* _WIN32 */


        /**
         * Generates the next-smaller multiple of AllocationGranularity to make sure it is aligned
         * and can be used for view mapping.
         *
         * @param position the number to be aligned.
         *
         * @return the aligned number
         */
        inline File::FileSize AlignPosition(File::FileSize position);

        /**
         * Generates a view on the current mapping of the open file, taking into account the current
         * view size as well as the actual file size. The start of the view is the aligned value of
         * this->filePos. this->viewStart is updated accordingly.
         * 
         * @return pointer to the memory where the view resides
         *
         * @throws IOException if view creation fails. Use GetLastError().
         * @throws IllegalStateException if the current mapping is invalid or the file is closed, or the access type is wrong
         */
        inline char* SafeMapView();

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
        bool CommonOpen(const File::AccessMode accessMode);

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
        // unnecessary
#endif /* _WIN32 */

        /**
         * paging mode, writing needs this parameter for each new mapping...
         */
#ifdef _WIN32
        DWORD protect;
#else /* _WIN32 */
        int protect;
#endif

        /**
         * store the access mode for later reference (e.g. views)
         */
        File::AccessMode access;

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

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MEMMAPPED_FILE_H_INCLUDED */
