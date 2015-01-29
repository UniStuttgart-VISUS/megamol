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

#include "mmcore/ObjectDescriptionManager.h"
#include "mmcore/CallDescription.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {


    /** exporting template specialization */
    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API ObjectDescriptionManager<CallDescription>;


    /**
     * Class of rendering graph call description manager
     */
    class MEGAMOLCORE_API CallDescriptionManager
        : public ObjectDescriptionManager<CallDescription> {
    public:

        /** Type to iterate descriptions */
        typedef ObjectDescriptionManager<CallDescription>::DescriptionIterator DescriptionIterator;

        /**
         * Returns the only instance of this class.
         *
         * @return The only instance of this class.
         */
        static CallDescriptionManager * Instance();

        /**
         * Shuts the one and only instance
         */
        static void ShutdownInstance(void);

        /**
         * Assignment crowbar
         *
         * @param tar The targeted object
         * @param src The source object
         */
        void AssignmentCrowbar(Call *tar, Call *src);

        /** Private dtor. */
        virtual ~CallDescriptionManager(void);

    private:

        /** The one an only instance */
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        static vislib::SmartPtr<CallDescriptionManager> inst;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /**
         * Registers object in the one and only instance
         *
         * @param instance The one an only instance
         */
        static void registerObjects(CallDescriptionManager *instance);

        /** Private ctor. */
        CallDescriptionManager(void);

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
