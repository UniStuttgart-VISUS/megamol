/*
 * IPHostEntry.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPHOSTENTRY_H_INCLUDED
#define VISLIB_IPHOSTENTRY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Array.h"


namespace vislib {
namespace net {


    /**
     * The IPHostEntry class is used as a helper to return answers from DNS. It
     * associates DNS host names with their respective IP addresses.
     */
    class IPHostEntry {

    public:

        /** Ctor. */
        IPHostEntry(void);

        IPHostEntry(const IPHostEntry& rhs);

        /** Dtor. */
        ~IPHostEntry(void);

        IPHostEntry& operator =(const IPHostEntry& rhs);

    private:

        /* Allow DNS creating IPHostEntry objects with actual data. */
        friend class DNS;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPHOSTENTRY_H_INCLUDED */

