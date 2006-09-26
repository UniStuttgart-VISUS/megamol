/*
 * StdOutStream.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STDOUTSTREAM_H_INCLUDED
#define VISLIB_STDOUTSTREAM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/OutStream.h"


namespace vislib {
namespace sys {

    /**
     * Abstract interface for output streams.
     */
    class StdOutStream : public OutStream {
    public:

        /**
         * Tries to writes size bytes from buffer to the stream
         *
         * @param buffer The pointer to the data to be output
         * @param size   The number of bytes of the data
         *
         * @return The number of bytes successfully written to the stream
         */
        virtual EXTENT Write(void *buffer, EXTENT size);
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_STDOUTSTREAM_H_INCLUDED */
