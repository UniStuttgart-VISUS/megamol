/*
 * ASCIIFileBuffer.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ASCIIFileBuffer.h"
#include "vislib/assert.h"
#include "vislib/memutils.h"


/*
 * vislib::sys::ASCIIFileBuffer::ASCIIFileBuffer
 */
vislib::sys::ASCIIFileBuffer::ASCIIFileBuffer(void) : buffer(NULL), lines() {
    // Intentionally empty
}


/*
 * vislib::sys::ASCIIFileBuffer::~ASCIIFileBuffer
 */
vislib::sys::ASCIIFileBuffer::~ASCIIFileBuffer(void) {
    this->Clear();
    ASSERT(this->lines.Count() == 0);
    ASSERT(this->buffer == NULL);
}


/*
 * vislib::sys::ASCIIFileBuffer::Clear
 */
void vislib::sys::ASCIIFileBuffer::Clear(void) {
    this->lines.Clear(); // DO NOT DELETE individual pointers
    ARY_SAFE_DELETE(this->buffer);
}


/*
 * vislib::sys::ASCIIFileBuffer::LoadFile
 */
bool vislib::sys::ASCIIFileBuffer::LoadFile(vislib::sys::File& file) {
    SIZE_T l = static_cast<SIZE_T>(file.GetSize());
    file.SeekToBegin();

    this->Clear();

    // IO
    this->buffer = new char[l + 1];
    this->buffer[l] = 0;
    if (this->buffer == NULL) {
        throw vislib::Exception("Cannot allocate memory to store file",
            __FILE__, __LINE__);
    }
    SIZE_T rl = static_cast<SIZE_T>(file.Read(this->buffer, l));
    if (rl != l) {
        if (rl == 0) {
            return false; // cannot read from file; "file.read" should have
                          // thrown an exception
        }
        l = rl;
    }

    // line seeks
    const unsigned int PAGE_SIZE = 1024 * 1024;
    char *lsp = this->buffer;
    for (char *p = this->buffer; l > 0; l--, p++) {
        if ((p[0] == '\n') || (p[0] == '\r')) {
            // store lsp
            if (this->lines.Capacity() == this->lines.Count()) {
                this->lines.Resize(this->lines.Capacity() + PAGE_SIZE);
            }
            this->lines.Append(lsp);

            // prepare next
            if ((l > 1) && (p[1] != p[0])
                    && ((p[1] == '\n') || (p[1] == '\r'))) {
                p++;
                l--;
            }
            p[0] = '\0';
            lsp = p + 1;
        }
    }
    // no need for paging here
    this->lines.Append(lsp);
    this->lines.Trim();

    return true;
}
