/*
 * Writable.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_WRITABLE_H_INCLUDED
#define VISLIB_WRITABLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {

    /**
     * This class defines an interface for all classes that allow writing to, 
     * e. g. Files and Writer classes.
     */
    class Writable {

    public:

        /**
         * Write 'bufSize' bytes from 'buf' to the object.
         *
         * @param buf     A pointer to the data to be written.
         * @param bufSize The number of bytes to be written.
         *
         * @return The number of bytes actually written.
         *
         * @throws Exception An Exception or derived class can be thrown in case
         *                   of an severe error while writing.
         */
        virtual EXTENT Write(const void *buf, const EXTENT bufSize) = 0;
    };
}

#endif /* VISLIB_WRITABLE_H_INCLUDED */
