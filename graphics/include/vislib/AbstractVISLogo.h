/*
 * AbstractVISLogo.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVISLOGO_H_INCLUDED
#define VISLIB_ABSTRACTVISLOGO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace graphics {


    /**
     * This class provides a VIS logo object for test renderings.
     */
    class AbstractVISLogo {

    public:

        /** Dtor. */
        virtual ~AbstractVISLogo(void);

        /**
         * Create all required resources for rendering a VIS logo.
         *
         * @throws Exception In case of an error.
         */
        virtual void Create(void) = 0;

        /**
         * Render the VIS logo. Create() must have been called before.
         *
         * @throws Exception In case of an error.
         */
        virtual void Draw(void) = 0;

        /**
         * Release all resources of the VIS logo.
         *
         * @throws Exception In case of an error.
         */
        virtual void Release(void) = 0;

    protected:

        /** Ctor. */
        AbstractVISLogo(void);

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTVISLOGO_H_INCLUDED */
