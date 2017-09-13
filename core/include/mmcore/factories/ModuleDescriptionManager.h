/*
 * ModuleDescriptionManager.h
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace factories {

    /** exporting template specialization */
    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API ObjectDescriptionManager<ModuleDescription>;

    /**
     * Class of rendering graph module description manager
     */
    class MEGAMOLCORE_API ModuleDescriptionManager : public ObjectDescriptionManager<ModuleDescription> {
    public:

        /** ctor */
        ModuleDescriptionManager();

        /** dtor */
        virtual ~ModuleDescriptionManager();

        /**
         * Registers a module description
         *
         * @param Cp The ModuleDescription class
         */
        template<class Cp>
        void RegisterDescription() {
            this->Register(std::make_shared<const Cp>());
        }

        /**
         * Registers a module using a module auto description
         *
         * @param Cp The Module class
         */
        template<class Cp>
        void RegisterAutoDescription() {
            this->RegisterDescription<ModuleAutoDescription<Cp> >();
        }

    private:

        /* deleted copy ctor */
        ModuleDescriptionManager(const ModuleDescriptionManager& src) = delete;

        /* deleted assignment operator */
        ModuleDescriptionManager& operator=(const ModuleDescriptionManager& rhs) = delete;

    };

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED */
