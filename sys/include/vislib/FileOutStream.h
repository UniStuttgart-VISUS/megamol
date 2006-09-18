/*
 * FileOutStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FILEOUTSTREAM_H_INCLUDED
#define VISLIB_FILEOUTSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/OutStream.h"
#include "vislib/File.h"


namespace vislib {
namespace sys {

    /**
     * interface for output streams for File objects.
     */
    class FileOutStream : public OutStream {
    public:

        /** 
         * ctor 
         *
         * Creates an OutStream interface object for a given File object.
         * The ownership of the File object is not affected, thus the caller
         * must ensure that the File object is valid as long as this 
         * FileOutStream object lives.
         *
         * @param file Pointer to a file. If file is NULL the Write will
         *             always return 0.
         */
        FileOutStream(File *file);

        /**
         * Tries to writes size bytes from buffer to the stream
         *
         * uses vislib::sys::File::Write
         *
         * @param buffer The pointer to the data to be output
         * @param size   The number of bytes of the data
         *
         * @return The number of bytes successfully written to the stream
         */
        virtual EXTENT Write(void *buffer, EXTENT size);

    private:

        /** file object the data will be written to */
        File *outFile;

    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_FILEOUTSTREAM_H_INCLUDED */
