/*
 * ASCIIFileBuffer.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ASCIIFileBuffer.h"
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer
 */
vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer() : cnt(0) {
    this->ptr.line = NULL;
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer
 */
vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer(
        const vislib::sys::ASCIIFileBuffer::LineBuffer& src) : cnt(0) {
    this->ptr.line = NULL;
    *this = src;
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::~LineBuffer
 */
vislib::sys::ASCIIFileBuffer::LineBuffer::~LineBuffer(void) {
    if ((this->cnt > 0) && (this->ptr.words != NULL)) {
        delete[] this->ptr.words;
        this->ptr.words = NULL;
        this->cnt = 0;
    }
}


/*
 * vislib::sys::ASCIIFileBuffer::Line::operator=
 */
vislib::sys::ASCIIFileBuffer::LineBuffer&
vislib::sys::ASCIIFileBuffer::LineBuffer::operator=(
        const vislib::sys::ASCIIFileBuffer::LineBuffer& rhs) {
    if (this == &rhs) return *this;

    if ((this->cnt > 0) && (this->ptr.words != NULL)) {
        delete[] this->ptr.words;
        this->ptr.words = NULL;
    }

    this->cnt = rhs.cnt;
    if (this->cnt == 0) {
        this->ptr.line = rhs.ptr.line;
    } else {
        this->ptr.words = new char*[this->cnt];
        ::memcpy(this->ptr.words, rhs.ptr.words, sizeof(char*) * this->cnt);
    }

    return *this;
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer
 */
vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer(char *line) : cnt(0) {
    this->ptr.line = line;
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer
 */
vislib::sys::ASCIIFileBuffer::LineBuffer::LineBuffer(
        vislib::Array<char *>& words) : cnt(words.Count()) {
    if (this->cnt == 0) {
        this->ptr.line = NULL;
    } else {
        this->ptr.words = new char*[this->cnt];
        ::memcpy(this->ptr.words, words.PeekElements(),
            sizeof(char *) * this->cnt);
    }
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::operator=
 */
vislib::sys::ASCIIFileBuffer::LineBuffer&
vislib::sys::ASCIIFileBuffer::LineBuffer::operator=(char * line) {

    if ((this->cnt > 0) && (this->ptr.words != NULL)) {
        delete[] this->ptr.words;
        this->ptr.words = NULL;
    }

    this->cnt = 0;
    this->ptr.line = line;

    return *this;
}


/*
 * vislib::sys::ASCIIFileBuffer::LineBuffer::operator=
 */
vislib::sys::ASCIIFileBuffer::LineBuffer&
vislib::sys::ASCIIFileBuffer::LineBuffer::operator=(vislib::Array<char *>& words) {

    if ((this->cnt > 0) && (this->ptr.words != NULL)) {
        delete[] this->ptr.words;
        this->ptr.words = NULL;
    }

    this->cnt = words.Count();
    if (this->cnt == 0) {
        this->ptr.line = NULL;
    } else {
        this->ptr.words = new char*[this->cnt];
        ::memcpy(this->ptr.words, words.PeekElements(),
            sizeof(char*) * this->cnt);
    }

    return *this;
}



/*
 * vislib::sys::ASCIIFileBuffer::ASCIIFileBuffer
 */
vislib::sys::ASCIIFileBuffer::ASCIIFileBuffer(ParsingElement elements)
        : buffer(NULL), lines(), defElements(elements) {
    if (this->defElements == PARSING_DEFAULT) {
VLTRACE(VISLIB_TRCELVL_WARN, "ASCIIFileBuffer(PARSING_DEFAULT) illegal\n");
        this->defElements = PARSING_LINES;
    }
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
bool vislib::sys::ASCIIFileBuffer::LoadFile(vislib::sys::File& file,
        ParsingElement elements) {
    if (elements == PARSING_DEFAULT) {
        elements = this->defElements;
    }
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

    // tokens seeks
    const unsigned int PAGE_SIZE = 1024 * 1024;

    if (elements == PARSING_LINES) {
        // line seeks
        char *lsp = this->buffer; // line start pointer
        for (char *p = this->buffer; l > 0; l--, p++) {
            if ((p[0] == '\n') || (p[0] == '\r')) {
                // store lsp
                if (this->lines.Capacity() == this->lines.Count()) {
                    this->lines.Resize(this->lines.Capacity() + PAGE_SIZE);
                }
                this->lines.Append(lsp);

                p[0] = '\0';
                // prepare next
                if ((l > 1) && (p[1] != p[0])
                        && ((p[1] == '\n') || (p[1] == '\r'))) {
                    p++;
                    l--;
                }
                lsp = p + 1;
            }
        }
        // no need for paging here
        this->lines.Append(lsp);

    } else if (elements == PARSING_WORDS) {
        // word seeks
        char *lsp = this->buffer; // line start pointer
        char *wsp = NULL;
        Array<char *> words;

        for (char *p = this->buffer; l > 0; l--, p++) {
            if (
#ifdef _USE_ISSPACE
                CharTraitsA::IsSpace(*p)
#else /* _USE_ISSPACE */
                (*p == ' ') || (*p == '\t')
#endif /* _USE_ISSPACE */
                    ) {
                if (wsp != NULL) {
                    words.Append(wsp);
                    wsp = NULL;
                    p[0] = '\0';
                }

            } else if ((p[0] == '\n') || (p[0] == '\r')) {
                // store lsp
                if (this->lines.Capacity() == this->lines.Count()) {
                    this->lines.Resize(this->lines.Capacity() + PAGE_SIZE);
                }
                if (wsp != NULL) {
                    words.Append(wsp);
                    wsp = NULL;
                }
                this->lines.Append(words);
                words.Clear();

                p[0] = '\0';
                // prepare next
                if ((l > 1) && (p[1] != p[0])
                        && ((p[1] == '\n') || (p[1] == '\r'))) {
                    p++;
                    l--;
                }
                lsp = p + 1;

            } else {
                if (wsp == NULL) {
                    wsp = p;
                }
            }
        }
        if (wsp != NULL) {
            words.Append(wsp);
        }
        // no need for paging here
        this->lines.Append(words);
        words.Clear();

    } else {
        throw vislib::IllegalStateException("ParsingElements illegal value",
            __FILE__, __LINE__);

    }

    this->lines.Trim();

    return true;
}
