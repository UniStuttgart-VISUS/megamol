/*
 * RawStorageWriter.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/RawStorageWriter.h"


/*
 * vislib::RawStorageWriter::RawStorageWriter
 */
vislib::RawStorageWriter::RawStorageWriter(vislib::RawStorage &store,
        size_t pos, size_t end, size_t inc) : data(store), end(end), inc(inc),
        pos(pos) {
    if (this->end == SIZE_MAX) {
        this->end = this->data.GetSize();
    }
    if (this->end < this->pos) {
        this->pos = this->end;
    }
    this->assertSize(this->end);
    if (this->inc == 0) {
        this->inc = RawStorageWriter::DEFAULT_INCREMENT;
    }
}


/*
 * vislib::RawStorageWriter::~RawStorageWriter
 */
vislib::RawStorageWriter::~RawStorageWriter(void) {
    // Intentionally empty
}


/*
 * vislib::RawStorageWriter::SetEnd
 */
void vislib::RawStorageWriter::SetEnd(size_t end) {
    this->end = end;
    if (this->end < this->pos) {
        this->pos = this->end;
    }
    this->assertSize(this->end);
}

/*
 * vislib::RawStorageWriter::SetIncrement
 */
void vislib::RawStorageWriter::SetIncrement(size_t inc) {
    this->inc = inc;
    if (this->inc == 0) {
        this->inc = RawStorageWriter::DEFAULT_INCREMENT;
    }
}


/*
 * vislib::RawStorageWriter::SetPosition
 */
void vislib::RawStorageWriter::SetPosition(size_t pos) {
    this->pos = pos;
    if (this->end < this->pos) {
        this->end = this->pos;
    }
    this->assertSize(this->end);
}


/*
 * vislib::RawStorageWriter::Write
 */
void vislib::RawStorageWriter::Write(const void *buf, size_t size) {
    this->assertSize(this->pos + size);
    ::memcpy(this->data.At(this->pos), buf, size);
    this->pos += size;
    if (this->end < this->pos) {
        this->end = this->pos;
    }
}


/*
 * vislib::RawStorageWriter::assertSize
 */
void vislib::RawStorageWriter::assertSize(size_t e) {
    if (e > this->data.GetSize()) {
        size_t s = this->data.GetSize();
        size_t d = e - s;
        size_t f = d / this->inc;
        if (d % this->inc) f++;
        d = f * this->inc;
        s = e + d;
        this->data.AssertSize(s, true);
    }
}
