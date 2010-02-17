/*
 * AbstractView3D.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTVIEW3D_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTVIEW3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/AbstractView.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of 3d rendering views
     */
    class AbstractView3D : public AbstractView {
    public:

        /** Ctor. */
        AbstractView3D(void);

        /** Dtor. */
        virtual ~AbstractView3D(void);

    protected:

    private:

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTVIEW3D_H_INCLUDED */
