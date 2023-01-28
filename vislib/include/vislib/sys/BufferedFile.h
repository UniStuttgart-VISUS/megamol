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


#include "vislib/sys/File.h"


namespace vislib::sys {

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
    inline static File::FileSize GetDefaultBufferSize() {
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
    BufferedFile();

    /**
     * Dtor. If the file is still open, it is closed.
     */
    ~BufferedFile() override;

    /** Close the file, if open. */
    void Close() override;

    /**
     * behaves like File::Flush
     *
     * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
     *                    could not be flushed to disk.
     */
    void Flush() override;

    /**
     * Answer the size of the current buffer in bytes.
     * The number of bytes of valid data in this buffer can differ.
     *
     * @return number of bytes of current buffer
     */
    inline File::FileSize GetBufferSize() const {
        return this->bufferSize;
    }

    /**
     * behaves like File::GetSize
     */
    File::FileSize GetSize() const override;

    /**
     * behaves like File::Open
     */
    bool Open(const char* filename, const File::AccessMode accessMode, const File::ShareMode shareMode,
        const File::CreationMode creationMode) override;

    /**
     * behaves like File::Open
     */
    bool Open(const wchar_t* filename, const File::AccessMode accessMode, const File::ShareMode shareMode,
        const File::CreationMode creationMode) override;

    /**
     * behaves like File::Read
     * Performs an implicite flush if the buffer is not in read mode.
     * Ensures that the buffer is in read mode.
     */
    File::FileSize Read(void* outBuf, const File::FileSize bufSize) override;

    /**
     * behaves like File::Seek
     * Performs an implicite flush if the buffer is in write mode.
     */
    File::FileSize Seek(const File::FileOffset offset, const File::SeekStartPoint from = File::BEGIN) override;

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
    File::FileSize Tell() const override;

    /**
     * behaves like File::Write
     * Performs an implicite flush if the buffer is not in write mode.
     * Ensures that the buffer is in write mode.
     *
     * throws IOException with ERROR_WRITE_FAULT if a buffer in write mode
     *                    could not be flushed to disk.
     */
    File::FileSize Write(const void* buf, const File::FileSize bufSize) override;

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
    BufferedFile& operator=(const BufferedFile& rhs);

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
    unsigned char* buffer;

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

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BUFFERED_FILE_H_INCLUDED */
