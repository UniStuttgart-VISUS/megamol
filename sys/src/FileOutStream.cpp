/*
 * FileOutStream.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/FileOutStream.h"


/*
 * vislib::sys::FileOutStream::FileOutStream
 */
vislib::sys::FileOutStream::FileOutStream(File *file) : outFile(file) {
}


/*
 * vislib::sys::FileOutStream::Write
 */
EXTENT vislib::sys::FileOutStream::Write(void *buffer, EXTENT size) {
    return (this->outFile) ? this->outFile->Write(buffer, size) : 0;
}
