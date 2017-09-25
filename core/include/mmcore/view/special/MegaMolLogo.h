/*
 * MegaMolLogo.h
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED
#define MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/graphics/AbstractVISLogo.h"


namespace megamol {
namespace core {
namespace view {
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

        /**
         * Answer the maximum x coordinate
         *
         * @return The maximum x coordinate
         */
        inline float MaxX(void) const {
            return this->maxX;
        }

    private:

        /** vertex pointer */
        float *vertices;

        /** colour pointer */
        unsigned char *colours;

        /** The triangles index pointer */
        unsigned int *triIdxs;

        /** number of triangle indices*/
        unsigned int triIdxCnt;

        /** The lines index pointer */
        unsigned int *lineIdxs;

        /** number of lines indices*/
        unsigned int lineIdxCnt;

        /** The maximum x coordinate */
        float maxX;

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MEGAMOLLOGO_H_INCLUDED */
