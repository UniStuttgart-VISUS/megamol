/*
 * MemoryFile.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/MemoryFile.h"

using namespace vislib::sys;


/*
 * MemoryFile::MemoryFile
 */
MemoryFile::MemoryFile(void) : File(), accessMode(READ_ONLY), buffer(NULL),
        bufferLen(0), pos(0), storage(NULL) {
}


/*
 * MemoryFile::~MemoryFile
 */
MemoryFile::~MemoryFile(void) {
    this->Close();
}


/*
 * MemoryFile::Close
 */
void MemoryFile::Close(void) {
    this->accessMode = READ_ONLY;
    this->buffer = NULL; // DO NOT DELETE
    this->bufferLen = 0;
    this->pos = 0;
    this->storage = NULL; // DO NOT DELETE
}


/*
 * MemoryFile::Flush
 */
void MemoryFile::Flush(void) {
    // Intentionally empty as memory files have no additional buffering
}


/*
 * MemoryFile::GetSize
 */
File::FileSize MemoryFile::GetSize(void) const {
    return (this->storage != NULL)
        ? static_cast<File::FileSize>(this->storage->GetSize())
        : ((this->buffer != NULL) ? this->bufferLen : 0);
}


/*
 * MemoryFile::IsOpen
 */
bool MemoryFile::IsOpen(void) const {
    return (this->storage != NULL) || (this->buffer != NULL);
}


/*
 * MemoryFile::Open
 */
bool MemoryFile::Open(void *buffer, SIZE_T bufferLength,
        File::AccessMode accessMode) {
    if (buffer == NULL) return false;
    this->Close();
    this->accessMode = accessMode;
    this->buffer = static_cast<unsigned char*>(buffer);
    this->bufferLen = bufferLength;
    this->pos = 0;
    this->storage = NULL; // paranoia
    return true;
}


/*
 * MemoryFile::Open
 */
bool MemoryFile::Open(vislib::RawStorage& storage,
        File::AccessMode accessMode) {
    this->Close();
    this->accessMode = accessMode;
    this->buffer = NULL; // paranoia
    this->bufferLen = 0; // paranoia
    this->pos = 0;
    this->storage = &storage;
    return true;
}


/*
 * MemoryFile::Open
 */
bool MemoryFile::Open(const char *filename, const File::AccessMode accessMode,
        const File::ShareMode shareMode,
        const File::CreationMode creationMode) {
    // Intentionally empty as memory files cannot open real files
    return false;
}


/*
 * MemoryFile::Open
 */
bool MemoryFile::Open(const wchar_t *filename,
        const File::AccessMode accessMode, const File::ShareMode shareMode,
        const File::CreationMode creationMode) {
    // Intentionally empty as memory files cannot open real files
    return false;
}


/*
 * MemoryFile::Read
 */
File::FileSize MemoryFile::Read(void *outBuf, const File::FileSize bufSize) {
    FileSize s = bufSize;
    FileSize l = this->GetSize();

    if (this->accessMode == WRITE_ONLY) return 0;

    if (this->pos > l) {
        this->pos = l;
    }
    if (this->pos + s > l) {
        s = l - this->pos;
    }
    if (!this->IsOpen()) {
        s = 0;
    }
    if (s > 0) {
        ::memcpy(outBuf, (this->storage != NULL)
            ? this->storage->At(static_cast<SIZE_T>(this->pos))
            : (this->buffer + this->pos), static_cast<SIZE_T>(s));
        this->pos += s;
    }
    return s;
}


/*
 * MemoryFile::Seek
 */
File::FileSize MemoryFile::Seek(const File::FileOffset offset,
        const File::SeekStartPoint from) {
    FileSize np = this->pos;
    switch (from) {
        case BEGIN:
            if (offset >= 0) {
                np = offset;
            }
            break;
        case CURRENT:
            if (offset < 0) {
                if (static_cast<FileSize>(-offset) > this->pos) {
                    np = 0;
                } else {
                    np = this->pos + offset;
                }
            } else {
                np = this->pos + offset;
            }
            break;
        case END:
            np = this->GetSize();
            if (offset < 0) {
                if (static_cast<FileSize>(-offset) > np) {
                    np = 0;
                } else {
                    np += offset;
                }
            }
            break;
    }
    if (np > this->GetSize()) {
        np = this->GetSize();
    }
    this->pos = np;
    return this->pos;
}


/*
 * MemoryFile::Tell
 */
File::FileSize MemoryFile::Tell(void) const {
    return this->pos;
}


/*
 * MemoryFile::Write
 */
File::FileSize MemoryFile::Write(const void *buf,
        const File::FileSize bufSize) {
    FileSize s = 0;

    if (this->accessMode == READ_ONLY) return 0;

    if (this->buffer != NULL) {
        s = bufSize;
        if (this->pos + s > this->bufferLen) {
            if (this->pos > this->bufferLen) {
                this->pos = this->bufferLen;
            }
            s = this->bufferLen - this->pos;
        }
        ::memcpy(this->buffer + this->pos, buf, static_cast<SIZE_T>(s));
        this->pos += s;

    } else if (this->storage != NULL) {
        s = bufSize;
        if (this->storage->GetSize() < this->pos + s) {
            // growing by size could be nice here, but ATM I don't care
            this->storage->AssertSize(static_cast<SIZE_T>(this->pos + s),
                true);
            if (this->pos > this->storage->GetSize()) {
                this->pos = this->storage->GetSize();
            }
            s = this->storage->GetSize() - this->pos;
        }
        ::memcpy(this->storage->At(static_cast<SIZE_T>(this->pos)), buf,
            static_cast<SIZE_T>(s));
        this->pos += s;

    } else {
        s = 0; // file not open
    }

    return s;
}
