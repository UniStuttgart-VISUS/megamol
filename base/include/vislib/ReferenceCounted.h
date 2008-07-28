/*
 * ReferenceCounted.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REFERENCECOUNTED_H_INCLUDED
#define VISLIB_REFERENCECOUNTED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/array.h"
#include "vislib/SingleAllocator.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This class adds a reference counting mechanism to a class.
     *
     * The template parameter A is the allocator used for releasing the
     * object once its reference count falls to zero.
     */
    template<class A> class ReferenceCounted {

    public:

        inline UINT32 AddRef(void) {
            return ++this->cntRefs;
        }

        UINT32 Release(void);

    protected:

        /** Ctor. */
        ReferenceCounted(void);

        /** Dtor. */
        virtual ~ReferenceCounted(void);

    private:
        
        /** The current reference count. */
        UINT32 cntRefs;

    };


    /*
     * vislib::ReferenceCounted::Release
     */
    UINT32 ReferenceCounted::Release(void) {
        UINT32 retval = --this->cntRefs;
        if (this->cntRefs == 0) {
            A::Deallocate(this);
        }
        return retval;
    }


    /*
     * vislib::ReferenceCounted::ReferenceCounted
     */
    ReferenceCounted::ReferenceCounted(void) {
    }


    /*
     * vislib::ReferenceCounted::~ReferenceCounted
     */
    ReferenceCounted::~ReferenceCounted(void) {
    }

    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REFERENCECOUNTED_H_INCLUDED */

