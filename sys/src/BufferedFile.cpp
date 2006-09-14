/*
 * BufferedFile.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/BufferedFile.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
//#include "vislib/IllegalStateException.h"
#include "vislib/IOException.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::sys::BufferedFile::defBufferSize
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::defaultBufferSize = 1024;


/*
 * vislib::sys::BufferedFile::BufferedFile
 */
vislib::sys::BufferedFile::BufferedFile(void) 
        : File(), buffer(NULL), bufferStart(0), bufferSize(BufferedFile::defaultBufferSize), 
        bufferMode(BufferedFile::VOID_BUFFER), bufferOffset(0), validBufferSize(0) {
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
    this->Flush();
    File::Close();
}


/*
 * vislib::sys::BufferedFile::Flush
 */
void vislib::sys::BufferedFile::Flush(void) {
    switch (this->bufferMode) {
        case VOID_BUFFER: /* no action needed */ break;
        case READ_BUFFER: {
            File::Seek(this->Tell(), vislib::sys::File::BEGIN); // move real file ptr to virtual pos
            this->validBufferSize = 0; // invalidate buffer
            this->bufferStart = File::Tell();
            this->bufferOffset = 0;
        } break;
        case WRITE_BUFFER: {
            vislib::sys::File::FileSize s = File::Write(this->buffer, this->bufferOffset); // flush write buffer; also corrects real file ptr pos
            if (s != this->bufferOffset) {
#ifdef _WIN32
                throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
#else /* _WIN32 */
                throw IOException(EIO, __FILE__, __LINE__);                
#endif /* _WIN32 */
            }
            this->validBufferSize = 0; // invalidate buffer
            this->bufferStart = File::Tell();
            this->bufferOffset = 0;
        } break;
        default: assert(false); // should never be called!
    }

    if (this->IsOpen()) {
        /* only flush opened files to avoid IOExceptions */
        File::Flush();
    }
}


/*
 * vislib::sys::BufferedFile::GetSize
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::GetSize(void) const {
    FileSize size = File::GetSize();
    if (this->bufferStart + this->bufferOffset > size) size = this->bufferStart + this->bufferOffset;
    return size;
}


/*
 * vislib::sys::BufferedFile::Open
 */
bool vislib::sys::BufferedFile::Open(const char *filename, const vislib::sys::File::AccessMode accessMode, 
        const vislib::sys::File::ShareMode shareMode, const vislib::sys::File::CreationMode creationMode) {
    if (File::Open(filename, accessMode, shareMode, creationMode)) {
        ARY_SAFE_DELETE(this->buffer);
        this->bufferMode = VOID_BUFFER;
        this->bufferStart = 0;
        this->validBufferSize = 0;
        this->bufferOffset = 0;
        return true;
    } else {
        return false;
    }
}


/*
 * vislib::sys::BufferedFile::Open
 */
bool vislib::sys::BufferedFile::Open(const wchar_t *filename, const vislib::sys::File::AccessMode accessMode, 
        const vislib::sys::File::ShareMode shareMode, const vislib::sys::File::CreationMode creationMode) {
    if (File::Open(filename, accessMode, shareMode, creationMode)) {
        ARY_SAFE_DELETE(this->buffer);
        this->bufferMode = VOID_BUFFER;
        this->bufferStart = 0;
        this->validBufferSize = 0;
        this->bufferOffset = 0;
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
    unsigned char *target = static_cast<unsigned char*>(outBuf);
    vislib::sys::File::FileSize size = bufSize;
    vislib::sys::File::FileSize read = 0;
    vislib::sys::File::FileSize s;

    if (this->bufferMode != READ_BUFFER) {
        // change buffer mode to read 
        if (this->bufferMode == WRITE_BUFFER) {
            this->Flush();
        }
        if (this->bufferMode == VOID_BUFFER) {
            // allocate memory 
            this->buffer = new unsigned char[static_cast<size_t>(this->bufferSize)];
        }
        ASSERT(this->validBufferSize == 0);
        ASSERT(this->bufferOffset == 0);
        this->bufferMode = READ_BUFFER;
    }

    while (size > 0) {
        if (this->validBufferSize > this->bufferOffset) {
            // buffered data available 
            s = this->validBufferSize - this->bufferOffset;
            if (s > size) s = size;
            memcpy(target, this->buffer + this->bufferOffset, static_cast<size_t>(s));
            this->bufferOffset += s;
            target += s;
            read += s;
            size -= s;
        } else {
            // no buffered data 
            if (size >= this->bufferSize) {
                // a really big block. Read it unbuffered
                s = File::Read(target, size);
                read += s;

                this->bufferStart += this->validBufferSize + s;
                this->bufferOffset = 0;
                this->validBufferSize = 0;
                ASSERT(this->bufferStart == File::Tell());
            } else {
                // fill the buffer 
                this->bufferStart += this->validBufferSize;
                this->bufferOffset = 0;
                this->validBufferSize = 0;
                ASSERT(this->bufferStart == File::Tell());

                this->validBufferSize = File::Read(this->buffer, this->bufferSize);

                if (this->validBufferSize < this->bufferSize) {
                    // eof encountered 
                    s = this->validBufferSize;
                    if (s > size) s = size;
                    memcpy(target, this->buffer, static_cast<size_t>(s));
                    this->bufferOffset += s;
                    read += s;
                    break; // we for sure stop here, because there is no more data
                }
            }
        }
    }

    return read;
}


/*
 * vislib::sys::BufferedFile::Seek
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Seek(const vislib::sys::File::FileOffset offset, 
        const vislib::sys::File::SeekStartPoint from) {
    // TODO: Consider only modifying bufferOffset when targeted possition is inside the buffer.
    //  Also consider rewriting Read and Write to minimize real flushes
    if (this->bufferMode == WRITE_BUFFER) {
        this->Flush();
    }
    if ((from == vislib::sys::File::CURRENT) && (this->bufferMode != VOID_BUFFER)) {
        return File::Seek(offset + this->Tell(), vislib::sys::File::BEGIN);
    } else {
        return File::Seek(offset, from);
    }
}


/*
 * vislib::sys::BufferedFile::SetBufferSize
 */
void vislib::sys::BufferedFile::SetBufferSize(vislib::sys::File::FileSize newSize) {
    this->Flush();
    ARY_SAFE_DELETE(this->buffer);

    this->bufferSize = newSize;

    this->bufferMode = VOID_BUFFER;
    this->bufferStart = 0;
    this->validBufferSize = 0;
    this->bufferOffset = 0;
}


/*
 * vislib::sys::BufferedFile::Tell
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Tell(void) const {
    return (this->bufferMode == VOID_BUFFER) ? File::Tell() : (this->bufferStart + this->bufferOffset);
}


/*
 * vislib::sys::BufferedFile::Write
 */
vislib::sys::File::FileSize vislib::sys::BufferedFile::Write(const void *buf, 
        const vislib::sys::File::FileSize bufSize) {
    unsigned char *source = static_cast<unsigned char *>(const_cast<void*>(buf));
    vislib::sys::File::FileSize size = bufSize;
    vislib::sys::File::FileSize written = 0;
    vislib::sys::File::FileSize s;

    if (this->bufferMode != WRITE_BUFFER) {
        // change buffer mode to write buffer
        if (this->bufferMode == READ_BUFFER) {
            this->Flush();
        }
        if (this->bufferMode == VOID_BUFFER) {
            // allocate memory 
            this->buffer = new unsigned char[static_cast<size_t>(this->bufferSize)];
        }
        ASSERT(this->validBufferSize == 0);
        ASSERT(this->bufferOffset == 0);
        this->bufferMode = WRITE_BUFFER;
    }

    while (size > 0) {
        if (this->bufferSize > this->bufferOffset) {
            // buffer is not full
            s = this->bufferSize - this->bufferOffset;
            if (s > size) s = size;
            memcpy(this->buffer, source, static_cast<size_t>(s));
            this->bufferOffset += s;
            this->validBufferSize = this->bufferSize;
            written += s;
            size -= s;
            source += s;
        }

        if (this->bufferOffset == this->bufferSize) {
            // buffer is full! flush it!
            s = File::Write(this->buffer, this->bufferSize);
            if (s != this->bufferSize) {
#ifdef _WIN32
                throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
#else /* _WIN32 */
                throw IOException(EIO, __FILE__, __LINE__);                
#endif /* _WIN32 */
            }
            this->bufferStart += this->bufferOffset;
            this->bufferOffset = this->validBufferSize = 0;

            if (size >= this->bufferSize) {
                // a big block. write at once 
                s = File::Write(source, size);
                written += s;
                this->bufferStart += s;
                break; // we're done here (whatever happend above)
            } // else continue normal work, since the buffer is empty now
        }
    }

    return written;
}


/*
 * vislib::sys::BufferedFile::BufferedFile copy ctor
 */
vislib::sys::BufferedFile::BufferedFile(const vislib::sys::BufferedFile& rhs) {
    throw UnsupportedOperationException("vislib::sys::File::File", __FILE__, __LINE__);
}


/*
 * vislib::sys::BufferedFile::operator =
 */
vislib::sys::BufferedFile& vislib::sys::BufferedFile::operator =(const vislib::sys::BufferedFile& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}
