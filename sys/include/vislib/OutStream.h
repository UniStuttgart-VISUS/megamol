/*
 * OutStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OUTSTREAM_H_INCLUDED
#define VISLIB_OUTSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Abstract interface for output streams.
     */
    class OutStream {
    public:

        /**
         * Tries to writes size bytes from buffer to the stream
         *
         * @param buffer The pointer to the data to be output
         * @param size   The number of bytes of the data
         *
         * @return The number of bytes successfully written to the stream
         */
        virtual EXTENT Write(void *buffer, EXTENT size) = 0;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_OUTSTREAM_H_INCLUDED */
