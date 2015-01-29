/*
 * ModuleNamespace.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED
#define MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractNamedObjectContainer.h"
#include "mmcore/param/AbstractParam.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {


    /**
     * Class represents a normal module namespace.
     */
    class MEGAMOLCORE_API ModuleNamespace: public AbstractNamedObjectContainer {
    public:

        /**
         * Ctor.
         *
         * @param name The name for the namespace
         */
        ModuleNamespace(const vislib::StringA& name);

        /**
         * Dtor.
         */
        virtual ~ModuleNamespace(void);

        /**
         * Clears the cleanup mark for this and all dependent objects.
         */
        virtual void ClearCleanupMark(void);

    protected:

    private:

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MODULENAMESPACE_H_INCLUDED */
