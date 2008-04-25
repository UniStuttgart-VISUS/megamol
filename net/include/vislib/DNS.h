/*
 * DNS.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DNS_H_INCLUDED
#define VISLIB_DNS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"
#include "vislib/IPHostEntry.h"
#include "vislib/String.h"


namespace vislib {
namespace net {


    /**
     * This class povides DNS queries via static methods.
     */
    class DNS {

    public:

        static IPHostEntry GetHostEntry(const IPAddress& hostAddress);

        static IPHostEntry GetHostEntry(const IPAddress6& hostAddress);

        static IPHostEntry GetHostEntry(const char *hostName);

        /** Dtor. */
        ~DNS(void);

     private:

        /** 
         * Disallow instances.
         */
        DNS(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        DNS(const DNS& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        DNS& operator =(const DNS& rhs);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DNS_H_INCLUDED */

