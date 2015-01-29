/*
 * AbstractRenderOutput.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTRENDEROUTPUT_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTRENDEROUTPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering output transportation classes to be used when building calls
     *
     * Note:
     *  This interface is used as root from multiple inheritance in diamond-constructs!
     * You must use virtual inheritance to correctly resolve all references to this class!
     */
    class MEGAMOLCORE_API AbstractRenderOutput {
    public:

        /**
         * Deactivates the output buffer
         */
        virtual void DisableOutputBuffer(void) = 0;

        /**
         * Activates the output buffer
         */
        virtual void EnableOutputBuffer(void) = 0;

    protected:

        /** Ctor */
        AbstractRenderOutput(void);

        /** Dtor */
        virtual ~AbstractRenderOutput(void);

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTRENDEROUTPUT_H_INCLUDED */
