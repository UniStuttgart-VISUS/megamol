/*
 * TextFileReader.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/TextFileReader.h"
#include "vislib/assert.h"
#include "vislib/UnsupportedOperationException.h"

#include "vislib/sysfunctions.h"


/*
 * vislib::sys::TextFileReader::TextFileReader
 */
vislib::sys::TextFileReader::TextFileReader(vislib::sys::File *file,
        unsigned int bufferSize) : buf(NULL), bufPos(0), bufSize(bufferSize),
        bufStart(0), file(file), validBufSize(0) {
    if (file != NULL) {
        this->FilePositionToReaderPosition();
    }
    this->buf = new char[this->bufSize];
}


/*
 * vislib::sys::TextFileReader::~TextFileReader
 */
vislib::sys::TextFileReader::~TextFileReader(void) {
    ARY_SAFE_DELETE(this->buf);
    this->validBufSize = 0;
    this->file = NULL; // DO NOT DELETE!
}


/*
 * vislib::sys::TextFileReader::FilePositionToReaderPosition
 */
void vislib::sys::TextFileReader::FilePositionToReaderPosition(void) {
    ASSERT(this->file != NULL);
    this->bufStart = this->file->Tell();
    this->bufPos = 0;
    this->validBufSize = 0;
}


/*
 * vislib::sys::TextFileReader::ReaderPositionToFilePosition
 */
void vislib::sys::TextFileReader::ReaderPositionToFilePosition(void) {
    ASSERT(this->file != NULL);
    this->file->Seek(this->bufStart + this->bufPos);
}


/*
 * vislib::sys::TextFileReader::ReadLine
 */
bool vislib::sys::TextFileReader::ReadLine(vislib::StringA& outLine,
        unsigned int maxSize) {
    ASSERT(this->file != NULL);

    vislib::StringA prepend;
    unsigned int start = this->bufPos;
    unsigned int maxToRead = maxSize;
    unsigned int len = 0;

    while (maxToRead > 0) {

        if (this->bufPos >= this->validBufSize) {
            if (len > 0) {
                // copy partial line to output variable
                prepend.Append(StringA(this->buf + start, len));
            }

            // buffer depleted. Need new data.
            this->bufStart += this->validBufSize;
            ASSERT(this->file->Tell() == this->bufStart); // inconsitency
            // ... detected, maybe we should always seek here (or at least
            // throw and exception instead of an assertion. Think about!
            this->bufPos = 0;
            start = 0;
            len = 0;
            this->validBufSize = static_cast<unsigned int>(
                this->file->Read(this->buf, this->bufSize));
            if (this->validBufSize == 0) {
                // unable to read, maybe eof
                outLine = prepend;
                return !outLine.IsEmpty();
            }
        }

        if ((this->buf[this->bufPos] == 0x0A)
                || (this->buf[this->bufPos] == 0x0D)) {
            // I detected a new line character :-)
            outLine = prepend + StringA(this->buf + start, len + 1);
            len = outLine.Length();
            // make the newline pretty
            outLine[static_cast<int>(len) - 1] = '\n';

            // check for combined newline
            char c1 = this->buf[this->bufPos++], c2 = 0;
            if (this->bufPos < this->validBufSize) {
                c2 = this->buf[this->bufPos];
            } else {
                // arglegarglgarg EndOfBuffer Buhuhu
                this->bufStart += this->validBufSize;
                ASSERT(this->file->Tell() == this->bufStart); // inconsitency
                // ... detected, maybe we should always seek here (or at least
                // throw and exception instead of an assertion. Think about!
                this->bufPos = 0;
                start = 0;
                len = 0;
                this->validBufSize = static_cast<unsigned int>(
                    this->file->Read(this->buf, this->bufSize));
                if (this->validBufSize > 0) {
                    c2 = this->buf[0];
                }
            }

            if (((c2 == 0x0A) || (c2 == 0x0D)) && (c2 != c1)) {
                // this is a combined newline, at least I am pretty sure
                this->bufPos++;
            }

            return true;

        } else {
            // normal character.
            this->bufPos++;
            len++;
            maxToRead--;
        }

    }

    // no new line buf read max characters.
    outLine = prepend + StringA(this->buf + start, len);
    return true;
}


/*
 * vislib::sys::TextFileReader::ReadLine
 */
bool vislib::sys::TextFileReader::ReadLine(vislib::StringW& outLine,
        unsigned int maxSize) {
    ASSERT(this->file != NULL);

    // Think about it! Reading unicode files seams kinda odd.
    // Would need encoding or something similar.

    vislib::StringA line;
    if (this->ReadLine(line, maxSize)) {
        outLine = line;
        return true;
    }

    return false;
}


/*
 * vislib::sys::TextFileReader::SetBufferSize
 */
void vislib::sys::TextFileReader::SetBufferSize(unsigned int bufferSize) {
    if (this->file != NULL) {
        this->ReaderPositionToFilePosition();
    }
    delete[] this->buf;
    this->bufSize = bufferSize;
    this->buf = new char[this->bufSize];
    this->bufPos = 0;
    this->validBufSize = 0;
    if (this->file != NULL) {
        this->FilePositionToReaderPosition();
    }
}


/*
 * vislib::sys::TextFileReader::SetFile
 */
void vislib::sys::TextFileReader::SetFile(File *file) {
    if (this->file != NULL) {
        this->ReaderPositionToFilePosition();
    }
    this->file = file;
    if (this->file != NULL) {
        this->FilePositionToReaderPosition();
    }
}


/*
 * vislib::sys::TextFileReader::TextFileReader
 */
vislib::sys::TextFileReader::TextFileReader(
        const vislib::sys::TextFileReader& src) {
    throw vislib::UnsupportedOperationException("TextFileReader::CopyCtor",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::TextFileReader::operator=
 */
vislib::sys::TextFileReader& vislib::sys::TextFileReader::operator=(
        const vislib::sys::TextFileReader& src) {
    if (&src != this) {
        throw vislib::UnsupportedOperationException(
            "TextFileReader::operator=", __FILE__, __LINE__);
    }
    return *this;
}
