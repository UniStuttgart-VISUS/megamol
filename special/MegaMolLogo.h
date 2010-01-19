/*
 * MegaMolLogo.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED
#define MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/AbstractVISLogo.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * Logo rendering helper class for the MegaMol™ logo
     */
    class MegaMolLogo : public vislib::graphics::AbstractVISLogo {
    public:

        /** Ctor. */
        MegaMolLogo(void);

        /** Dtor. */
        virtual ~MegaMolLogo(void);

        /**
         * Create all required resources for rendering a VIS logo.
         *
         * @throws Exception In case of an error.
         */
        virtual void Create(void);

        /**
         * Render the VIS logo. Create() must have been called before.
         *
         * @throws Exception In case of an error.
         */
        virtual void Draw(void);

        /**
         * Release all resources of the VIS logo.
         *
         * @throws Exception In case of an error.
         */
        virtual void Release(void);

    private:

        /** vertex pointer */
        void *vertices;

        /** colour pointer */
        void *colours;

        /** number of elements */
        unsigned int count;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED */
