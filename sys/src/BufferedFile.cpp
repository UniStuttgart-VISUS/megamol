/*
 * BufferedFile.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/BufferedFile.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IOException.h"
#include "vislib/memutils.h"
#include "vislib/SystemInformation.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/vislibsymbolimportexport.inl"

#ifdef _WIN32
#include <WinError.h>
#else /* _WIN32 */
#include <errno.h>
#endif /* _WIN32 */


/*
 * vislib::sys::BufferedFile::defBufferSize
 */
VISLIB_STATICSYMBOL vislib::sys::File::FileSize
__vl_bufferedfile_defaultBufferSize 
#ifndef VISLIB_SYMBOL_IMPORT
//    = vislib::sys::SystemInformation::PageSize();
    = 64 * 1024
#endif /* !VISLIB_SYMBOL_IMPORT */
    ;


/*
 * vislib::sys::BufferedFile::defBufferSize
 */
vislib::sys::File::FileSize& vislib::sys::BufferedFile::defaultBufferSize(
    __vl_bufferedfile_defaultBufferSize);


/*
 * vislib::sys::BufferedFile::BufferedFile
 */
vislib::sys::BufferedFile::BufferedFile(void)
        : File(), buffer(NULL), bufferOffset(0),
        bufferSize(BufferedFile::defaultBufferSize), bufferStart(0),
        dirtyBuffer(false), fileMode(File::READ_WRITE), validBufferSize(0) {
    this->buffer = new unsigned char[
        static_cast<unsigned int>(this->bufferSize)];
}


/*
 * vislib::sys::BufferedFile::~BufferedFile
 */
vislib::sys::BufferedFile::~BufferedFile(void) {
    ARY_SAFE_DELETE(this->buffer);
}


/*
 * vislib::sys::BufferedFile::Close
 */
void vislib::sys::BufferedFile::Close(void) {
    this->Flush(); // flushes if writeable and dirty.
    File::Close();
    this->fileMode = File::READ_WRITE;
    this->resetBuffer();
}


/*
 * vislib::sys::BufferedFile::Flush
 */
void vislib::sys::BufferedFile::Flush(void) {
    this->flush(true);
}


/*
 * vislib::sys::BufferedFile::GetSize
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::GetSize(void) const {
    FileSize size = File::GetSize();
    if (this->bufferStart + this->validBufferSize > size) {
        size = this->bufferStart + this->validBufferSize;
    }
    return size;
}


/*
 * vislib::sys::BufferedFile::Open
 */
bool vislib::sys::BufferedFile::Open(const char *filename,
        const vislib::sys::File::AccessMode accessMode,
        const vislib::sys::File::ShareMode shareMode,
        const vislib::sys::File::CreationMode creationMode) {

    if (File::Open(filename, accessMode, shareMode, creationMode)) {

        this->fileMode = accessMode;
        this->resetBuffer();

        return true;

    } else {
        return false;

    }
}


/*
 * vislib::sys::BufferedFile::Open
 */
bool vislib::sys::BufferedFile::Open(const wchar_t *filename,
        const vislib::sys::File::AccessMode accessMode,
        const vislib::sys::File::ShareMode shareMode,
        const vislib::sys::File::CreationMode creationMode) {

    if (File::Open(filename, accessMode, shareMode, creationMode)) {

        this->fileMode = accessMode;
        this->resetBuffer();

        return true;

    } else {
        return false;

    }
}


/*
 * vislib::sys::BufferedFile::Read
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Read(void *outBuf,
        const vislib::sys::File::FileSize bufSize) {
    // check for compatible file mode
    if ((this->fileMode != File::READ_WRITE) 
            && (this->fileMode != File::READ_ONLY)) {
        throw IOException(
#ifdef _WIN32
            E_ACCESSDENIED  /* access denied: wrong access mode */
#else /* _WIN32 */
            EINVAL          /* invalid file: file not open for read */
#endif /* _WIN32 */
            , __FILE__, __LINE__);
    }

    File::FileSize amount;
    unsigned char *outBuffer = static_cast<unsigned char *>(outBuf);
    File::FileSize outBufferSize = bufSize;

    // first copy from buffer
    amount = this->validBufferSize - this->bufferOffset;
    if (amount > outBufferSize) {
        amount = outBufferSize;
    }
    memcpy(outBuffer, this->buffer + this->bufferOffset,
        static_cast<unsigned int>(amount));
    this->bufferOffset += amount;
    outBufferSize -= amount;

    if (outBufferSize == 0) {
        return amount; // we are done!
    }

    outBuffer += amount;

    // buffer depleted. Must be replaced.
    this->flush(false);
    this->bufferStart += this->bufferOffset;
    this->validBufferSize = 0;
    this->bufferOffset = 0;

    File::FileSize read = amount;
    if (outBufferSize >= this->bufferSize) {
        // requesting large amounts, so read directly
        File::Seek(this->bufferStart, File::BEGIN);

        while (outBufferSize > 0) {
            amount = File::Read(outBuffer, outBufferSize);
            if (amount == 0) {
                break;
            }
            outBuffer += amount;
            outBufferSize -= amount;
            read += amount;
            this->bufferStart += amount;
        }

        return read;
    }

    // refill the buffer
    File::Seek(this->bufferStart, File::BEGIN);
    this->validBufferSize = File::Read(this->buffer, this->bufferSize);

    // read remaining data from the buffer
    amount = outBufferSize; // amount to read to fulfill the request.
    if (amount > this->validBufferSize) {
        amount = this->validBufferSize;
    }
    memcpy(outBuffer, this->buffer, static_cast<unsigned int>(amount));
    this->bufferOffset = amount;

    return read + amount;
}


/*
 * vislib::sys::BufferedFile::Seek
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Seek(
        const vislib::sys::File::FileOffset offset,
        const vislib::sys::File::SeekStartPoint from) {

    File::FileSize pos; // absolute file position to seek to
    switch (from) {
        case File::BEGIN :
            pos = offset;
            break;
        case File::CURRENT :
            pos = this->Tell() + offset;
            break;
        case File::END :
            pos = this->GetSize() + offset;
            break;
        default:
            throw vislib::IllegalParamException("from", __FILE__, __LINE__);
    }

    if ((pos >= this->bufferStart) 
            && (pos < this->bufferStart + this->validBufferSize)) {
        // we seek inside the buffer
        this->bufferOffset = pos - this->bufferStart;
        return pos;
    }

    // we cannot seek inside the buffer
    this->Flush();

    File::FileSize retVal = File::Seek(pos, File::BEGIN);
    this->validBufferSize = 0;
    this->bufferOffset = 0;
    this->bufferStart = retVal;

    return retVal;
}


/*
 * vislib::sys::BufferedFile::SetBufferSize
 */
void vislib::sys::BufferedFile::SetBufferSize(
        vislib::sys::File::FileSize newSize) {

    if (newSize < 1) {
        throw vislib::IllegalParamException("newSize", __FILE__, __LINE__);
    }

    this->Flush(); // writes buffer if dirty and file writeable
    this->resetBuffer();
    delete[] this->buffer;
    this->bufferSize = newSize;
    this->buffer = new unsigned char[
        static_cast<unsigned int>(this->bufferSize)];
}


/*
 * vislib::sys::BufferedFile::Tell
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Tell(void) const {
    return this->bufferStart + this->bufferOffset;
}


/*
 * vislib::sys::BufferedFile::Write
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Write(const void *buf,
        const vislib::sys::File::FileSize bufSize) {
    // check for compatible file mode
    if ((this->fileMode != File::READ_WRITE) 
            && (this->fileMode != File::WRITE_ONLY)) {
        throw IOException(
#ifdef _WIN32
            E_ACCESSDENIED  /* access denied: wrong access mode */
#else /* _WIN32 */
            EINVAL          /* invalid file: file not open for writing */
#endif /* _WIN32 */
            , __FILE__, __LINE__);
    }

    const unsigned char *inBuffer 
        = static_cast<const unsigned char *>(buf);
    File::FileSize inBufferSize = bufSize;
    File::FileSize amount;

    // store into buffer
    amount = this->bufferSize - this->bufferOffset;
    if (amount > inBufferSize) {
        amount = inBufferSize;
    }
    memcpy(this->buffer + this->bufferOffset, inBuffer,
        static_cast<unsigned int>(amount));
    inBufferSize -= amount;
    this->bufferOffset += amount;
    if (this->validBufferSize < this->bufferOffset) {
        this->validBufferSize = this->bufferOffset;
    }
    this->dirtyBuffer = true;
    if (inBufferSize == 0) {
        return amount; // we are done.
    }

    inBuffer += amount;

    // buffer full need to refresh
    this->flush(false);
    this->bufferStart += this->bufferOffset;
    this->validBufferSize = 0;
    this->bufferOffset = 0;

    if (inBufferSize > this->bufferSize) {
        // requesting large amounts, so write directly
        File::Seek(this->bufferStart, File::BEGIN);
        File::FileSize written = amount;

        while (inBufferSize > 0) {
            amount = File::Write(inBuffer, inBufferSize);
            if (amount == 0) {
                break;
            }
            inBuffer += amount;
            inBufferSize -= amount;
            written += amount;
            this->bufferStart += amount;
        }

        return written;
    }

    memcpy(this->buffer, inBuffer,
        static_cast<unsigned int>(inBufferSize));
    this->bufferOffset = inBufferSize;
    this->validBufferSize = inBufferSize;
    this->dirtyBuffer = true;

    return amount + inBufferSize;

}


/*
 * vislib::sys::BufferedFile::BufferedFile copy ctor
 */
vislib::sys::BufferedFile::BufferedFile(const vislib::sys::BufferedFile& rhs) {
    throw UnsupportedOperationException("vislib::sys::File::File",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::BufferedFile::operator =
 */
vislib::sys::BufferedFile& vislib::sys::BufferedFile::operator =(
        const vislib::sys::BufferedFile& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}


/*
 * vislib::sys::BufferedFile::flush
 */
void vislib::sys::BufferedFile::flush(bool fileFlush) {
    if (this->IsOpen() && ((this->fileMode == File::WRITE_ONLY)
            || (this->fileMode == File::READ_WRITE))) {

        if ((this->validBufferSize > 0) && this->dirtyBuffer) {

            File::Seek(this->bufferStart, File::BEGIN);
            File::FileSize w, r = 0;

            while (r < this->validBufferSize) {
                w = File::Write(this->buffer + r, this->validBufferSize + r);
                if (w == 0) {
#ifdef _WIN32
                    throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
#else /* _WIN32 */
                    throw IOException(EIO, __FILE__, __LINE__);
#endif /* _WIN32 */
                }
                r += w;
            }

        }

        if (fileFlush) {
            File::Flush();
        }
    }
    this->dirtyBuffer = false;
}


/*
 * vislib::sys::BufferedFile::resetBuffer
 */
void vislib::sys::BufferedFile::resetBuffer() {
    this->bufferOffset = 0;
    this->bufferStart = 0;
    this->dirtyBuffer = false;
    this->validBufferSize = 0;
}
