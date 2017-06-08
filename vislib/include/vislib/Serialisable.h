/*
 * Serialisable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SERIALISABLE_H_INCLUDED
#define VISLIB_SERIALISABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Serialiser.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This is the interface that classes must implement for being serialised
     * and deserialised using VISlib Serialisers.
     */
    class Serialisable {

    public:

        /** Dtor. */
        virtual ~Serialisable(void);

        /**
         * Deserialise the object from 'serialiser'. The caller must ensure that
         * the Serialiser is in an acceptable state to deserialise from.
         *
         * @param serialiser The Serialiser to deserialise the object from.
         *
         * @throws Exception Implementing classes may throw exceptions to 
         *                   indicate an error or pass through exceptions thrown
         *                   by the Serialiser.
         */
        virtual void Deserialise(Serialiser& serialiser) = 0;

        /**
         * Serialise the object to 'serialiser'. The caller must ensure that
         * the Serialiser is in an acceptable state to serialise to.
         *
         * @param serialiser The Serialiser to serialise the object to.
         *
         * @throws Exception Implementing classes may throw exceptions to 
         *                   indicate an error or pass through exceptions thrown
         *                   by the Serialiser.
         */
        virtual void Serialise(Serialiser& serialiser) const = 0;

    protected:

        /** Ctor. */
        Serialisable(void);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        Serialisable(const Serialisable& rhs);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Serialisable& operator =(const Serialisable& rhs);
    };
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERIALISABLE_H_INCLUDED */
