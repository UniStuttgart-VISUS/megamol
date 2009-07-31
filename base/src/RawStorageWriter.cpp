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
        SIZE_T pos, SIZE_T end, SIZE_T inc) : data(store), end(end), inc(inc),
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
void vislib::RawStorageWriter::SetEnd(SIZE_T end) {
    this->end = end;
    if (this->end < this->pos) {
        this->pos = this->end;
    }
    this->assertSize(this->end);
}

/*
 * vislib::RawStorageWriter::SetIncrement
 */
void vislib::RawStorageWriter::SetIncrement(SIZE_T inc) {
    this->inc = inc;
    if (this->inc == 0) {
        this->inc = RawStorageWriter::DEFAULT_INCREMENT;
    }
}


/*
 * vislib::RawStorageWriter::SetPosition
 */
void vislib::RawStorageWriter::SetPosition(SIZE_T pos) {
    this->pos = pos;
    if (this->end < this->pos) {
        this->end = this->pos;
    }
    this->assertSize(this->end);
}


/*
 * vislib::RawStorageWriter::Write
 */
void vislib::RawStorageWriter::Write(const void *buf, SIZE_T size) {
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
void vislib::RawStorageWriter::assertSize(SIZE_T e) {
    if (e > this->data.GetSize()) {
        SIZE_T s = this->data.GetSize();
        SIZE_T d = e - s;
        SIZE_T f = d / this->inc;
        if (d % this->inc) f++;
        d = f * this->inc;
        s = e + d;
        this->data.AssertSize(s, true);
    }
}
