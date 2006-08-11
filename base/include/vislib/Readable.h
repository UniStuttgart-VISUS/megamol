/*
 * Readable.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_READABLE_H_INCLUDED
#define VISLIB_READABLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {

    /**
     * Defines the interface for objects that can be read from like files.
     */
    class Readable {

    public:

        /**
         * Read at most 'bufSize' bytes from the object into 'outBuf'.
         *
         * @param outBuf  The buffer to read into.
         * @param bufSize The size of 'outBuf' in bytes.
         *
         * @return The number of bytes acutally read.
         *
         * @throws Exception An Exception or derived class can be thrown in case
         *                   of an severe error while writing.
         */
        virtual EXTENT Read(void *outBuf, const EXTENT bufSize) = 0;
    };
}

#endif /* VISLIB_READABLE_H_INCLUDED */
