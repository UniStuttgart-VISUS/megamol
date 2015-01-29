/*
 * ObjectDescription.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OBJECTDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_OBJECTDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"


namespace megamol {
namespace core {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class MEGAMOLCORE_API ObjectDescription {
    public:

        /**
         * Ctor.
         */
        ObjectDescription(void);

        /**
         * Dtor.
         */
        virtual ~ObjectDescription(void);

        /**
         * Answer the class name of the objects of this description.
         *
         * @return The class name of the objects of this description.
         */
        virtual const char *ClassName(void) const = 0;

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        virtual const char *Description(void) const = 0;

    };

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OBJECTDESCRIPTION_H_INCLUDED */
