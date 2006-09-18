/*
 * StdInStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STDINSTREAM_H_INCLUDED
#define VISLIB_STDINSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/InStream.h"


namespace vislib {
namespace sys {

    /**
     * Abstract interface for input streams.
     */
    class StdInStream : public InStream {
    public:

        /**
         * Tries to read size bytes from the stream to buffer.
         *
         * Since the data is read from the stdin stream, behaviour is depending
         * on the usage of stdin. If used for keyboard input, read will block
         * until size bytes are available (user normally has to confirm input
         * by hitting the enter key). If used by a pipe, read will return
         * immediately receiving size bytes or less, if the pipe reached it's 
         * end.
         *
         * @param buffer The pointer to the buffer receiving the data
         * @param size   The number of bytes to receive
         *
         * @return The number of bytes successfully read from the stream
         */
        virtual EXTENT Read(void *buffer, EXTENT size);
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_STDINSTREAM_H_INCLUDED */
