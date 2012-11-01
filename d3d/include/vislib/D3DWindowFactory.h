/*
 * D3DWindowFactory.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DWINDOWFACTORY_H_INCLUDED
#define VISLIB_D3DWINDOWFACTORY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * TODO: comment class
     */
    class D3DWindowFactory {

    public:

        /** Ctor. */
        D3DWindowFactory(void);

        /** Dtor. */
        ~D3DWindowFactory(void);

    protected:

    private:

    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DWINDOWFACTORY_H_INCLUDED */

