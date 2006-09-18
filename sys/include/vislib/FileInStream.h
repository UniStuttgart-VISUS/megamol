/*
 * FileInStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FILEINSTREAM_H_INCLUDED
#define VISLIB_FILEINSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/InStream.h"
#include "vislib/File.h"


namespace vislib {
namespace sys {

    /**
     * Interface for input streams for File objects.
     */
    class FileInStream : public InStream {
    public:

        /**
         * ctor
         *
         * Creates an IntStream interface object for a given File object.
         * The ownership of the File object is not affected, thus the caller
         * must ensure that the File object is valid as long as this 
         * FileInStream object lives.
         *
         * @param file Pointer to a file. If file is NULL the Read will
         *             always return 0.
         */
        FileInStream(File *file);

        /**
         * Tries to read size bytes from the stream to buffer.
         *
         * uses vislib::sys::File::Read
         *
         * @param buffer The pointer to the buffer receiving the data
         * @param size   The number of bytes to receive
         *
         * @return The number of bytes successfully read from the stream
         */
        virtual EXTENT Read(void *buffer, EXTENT size);

    private:

        /** file object the data will be read from */
        File *inFile;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_FILEINSTREAM_H_INCLUDED */
