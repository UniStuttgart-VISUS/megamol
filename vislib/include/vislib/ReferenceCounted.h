/*
 * ReferenceCounted.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REFERENCECOUNTED_H_INCLUDED
#define VISLIB_REFERENCECOUNTED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {


    /**
     * This class adds a reference counting mechanism to a class.
     *
     * Objects that inherit from ReferenceCounted must always be allocated on
     * the heap and must always use the C++ new operator for allocation, as 
     * ReferenceCounted will free itself using delete.
     *
     * The reference count of a newly created object is 1.
     */
    class ReferenceCounted {

    public:

        /**
         * Increment the reference count.
         *
         * @return The new value of the reference counter.
         */
        UINT32 AddRef(void);

        /**
         * Decrement the reference count. If the reference count reaches zero,
         * the object is released using the C++ delete operator.
         *
         * @return The new value of the reference counter.
         */
        UINT32 Release(void);

    protected:

        /** Ctor. */
        ReferenceCounted(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        ReferenceCounted(const ReferenceCounted& rhs);

        /** 
         * Dtor. 
         *
         * Making the dtor protected prevents explicit deletion of objects using
         * delete and creation of objects on the stack to a certain extent.
         */
        virtual ~ReferenceCounted(void);

        /**
         * Assignment.
         *
         * Note that assignment does not change the reference count of an
         * object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ReferenceCounted& operator =(const ReferenceCounted& rhs);

    private:
        
        /** The current reference count. */
        UINT32 cntRefs;
    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REFERENCECOUNTED_H_INCLUDED */
