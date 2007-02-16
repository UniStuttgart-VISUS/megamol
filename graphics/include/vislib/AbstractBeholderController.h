/*
 * AbstractBeholderController.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTBEHOLDERCONTROLLER_H_INCLUDED
#define VISLIB_ABSTRACTBEHOLDERCONTROLLER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace graphics {

    /* forward declaration */
    class Beholder;

    /**
     * Abstract interface class for beholder controller objects
     */
    class AbstractBeholderController {

    public:

        /** ctor */
        AbstractBeholderController(void);

        /** Dtor. */
        virtual ~AbstractBeholderController(void);

        /**
         * Associates a beholder with this beholder controller. The ownership
         * of the beholder is not changed, so the caller must ensure that the
         * beholder lives as long as it is associated with this controller.
         *
         * @param beholder The beholder.
         */
        void SetBeholder(Beholder *beholder);

        /**
         * Returns the associated beholder.
         *
         * @return The associated beholder.
         */
        inline Beholder * GetBeholder(void) {
            return this->beholder;
        }

    private:

        /** The beholder hold */
        Beholder *beholder;
    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTBEHOLDERCONTROLLER_H_INCLUDED */
