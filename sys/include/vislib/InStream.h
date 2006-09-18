/*
 * InStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_INSTREAM_H_INCLUDED
#define VISLIB_INSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Abstract interface for input streams.
     */
    class InStream {
    public:

        /**
         * Tries to read size bytes from the stream to buffer.
         *
         * @param buffer The pointer to the buffer receiving the data
         * @param size   The number of bytes to receive
         *
         * @return The number of bytes successfully read from the stream
         */
        virtual EXTENT Read(void *buffer, EXTENT size) = 0;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_INSTREAM_H_INCLUDED */
