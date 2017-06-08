/*
 * CallDescriptionManager.h
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/macro_utils.h"
#include "mmcore/Call.h"


namespace megamol {
namespace core {
namespace factories {

    /** exporting template specialization */
    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API ObjectDescriptionManager<CallDescription>;

    /**
     * Class of rendering graph call description manager
     */
    class MEGAMOLCORE_API CallDescriptionManager : public ObjectDescriptionManager<CallDescription> {
    public:

        /** Type to iterate descriptions */
        typedef ObjectDescriptionManager<CallDescription>::description_iterator_type description_iterator_type;
        typedef ObjectDescriptionManager<CallDescription>::description_const_iterator_type description_const_iterator_type;

    public:

        /** ctor */
        CallDescriptionManager();

        /** dtor */
        virtual ~CallDescriptionManager();

        /**
         * Registers a call description
         *
         * @param Cp The CallDescription class
         */
        template<class Cp>
        void RegisterDescription() {
            this->Register(std::make_shared<const Cp>());
        }

        /**
         * Registers a call using a module call description
         *
         * @param Cp The Call class
         */
        template<class Cp>
        void RegisterAutoDescription() {
            this->RegisterDescription<CallAutoDescription<Cp> >();
        }

        /**
         * Assignment crowbar
         *
         * @param tar The targeted object
         * @param src The source object
         *
         * @return True on success, false on failure.
         */
        bool AssignmentCrowbar(Call *tar, Call *src) const;

    private:

        /* deleted copy ctor */
        CallDescriptionManager(const CallDescriptionManager& src) = delete;

        /* deleted assignment operator */
        CallDescriptionManager& operator=(const CallDescriptionManager& rhs) = delete;

    };

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED */
