/*
 * CoreHandle.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_COREHANDLE_H_INCLUDED
#define MEGAMOLCON_COREHANDLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace console {

    /**
     * Wrapper class for MegaMol core and view handles
     */
    class CoreHandle {
    public:

        /**
         * ctor
         */
        CoreHandle(void);

        /**
         * dtor
         */
        ~CoreHandle(void);

        /** 
         * returns the handle pointer
         *
         * @return the handle pointer or NULL if no handle is associated
         */       
        operator void *(void) const;

        /**
         * Answer if the handle is valid
         *
         * @return 'true' if the handle is valid, 'false' otherwise.
         */
        inline bool IsValid(void) const {
            return ((this->hndl != NULL) && (this->hndl[0] != 0));
        }

        /**
         * Destroies the handle by disposing the handle and cleaning up the 
         * memory.
         */
        void DestroyHandle(void);

    private:

        /** pointer to the memory allocated */
        mutable unsigned char *hndl;

        /** the size of the memory allocated */
        mutable unsigned int size;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_COREHANDLE_H_INCLUDED */
