/*
 * CoreHandle.h
 *
 * Copyright (C) 2006 - 2016 MegaMol Team.
 * All rights reserved
 */

#ifndef MEGAMOLCON_COREHANDLE_H_INCLUDED
#define MEGAMOLCON_COREHANDLE_H_INCLUDED
#pragma once

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

        /** move ctor */
        CoreHandle(CoreHandle&& src);

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

        /** move operator */
        CoreHandle& operator=(CoreHandle&& src);

        /**
         * Answer if the handle is valid
         *
         * @return 'true' if the handle is valid, 'false' otherwise.
         */
        inline bool IsValid(void) const {
            return ((this->hndl != nullptr) && (this->hndl[0] != 0));
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
