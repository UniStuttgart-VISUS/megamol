/*
 * CallDescriptionManager.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_CALLDESCRIPTIONMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "ObjectDescriptionManager.h"
#include "CallDescription.h"


namespace megamol {
namespace core {


    /**
     * Class of rendering graph call description manager
     */
    class CallDescriptionManager
        : public ObjectDescriptionManager<CallDescription> {
    public:

        /**
         * Returns the only instance of this class.
         *
         * @return The only instance of this class.
         */
        static CallDescriptionManager * Instance();

    private:

        /** Private ctor. */
        CallDescriptionManager(void);

        /** Private dtor. */
        virtual ~CallDescriptionManager(void);

        /**
         * Registers a call description. Template parameter Cp is the call
         * description class.
         */
        template<class Cp>
        void registerDescription(void);

        /**
         * Registers a call description.
         *
         * @param desc The call description to be registered. The memory of
         *             the description object will be managed by the manager
         *             object. The caller must not alter the memory (e.g. free)
         *             after this call was issued.
         */
        void registerDescription(CallDescription* desc);

        /**
         * Registers a call using an auto-description object.
         * Template parameter Cp is the call class to be described by an
         * auto-description object.
         */
        template<class Cp>
        void registerAutoDescription(void);

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLDESCRIPTIONMANAGER_H_INCLUDED */
