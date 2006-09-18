/*
 * FileInStream.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/FileInStream.h"


/*
 * vislib::sys::FileInStream::FileInStream
 */
vislib::sys::FileInStream::FileInStream(File *file) : inFile(file) {
}


/*
 * vislib::sys::FileInStream::Read
 */
EXTENT vislib::sys::FileInStream::Read(void *buffer, EXTENT size) {
    return (this->inFile) ? this->inFile->Read(buffer, size) : 0;
}

