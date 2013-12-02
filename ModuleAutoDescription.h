/*
 * ModuleAutoDescription.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ModuleDescription.h"
#include "vislib/Log.h"
#include <typeinfo>


namespace megamol {
namespace core {


    /**
     * Class of rendering graph module descriptions generated using static
     * member implementations of the module classes.
     * Template parameter 'C' is the module class to be described.
     *
     * 'C' must implement the static methods:
     *      const char* ClassName();
     *      const char *Description();
     *      bool IsAvailable();
     */
    template<class C> class ModuleAutoDescription : public ModuleDescription {
    public:

        /** Ctor. */
        ModuleAutoDescription(void) : ModuleDescription() {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~ModuleAutoDescription(void) {
            // intentionally empty
        }

        /**
         * Answer the name of the module described.
         *
         * @return The name of the module described.
         */
        virtual const char *ClassName(void) const {
            return C::ClassName();
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        virtual const char *Description(void) const {
            return C::Description();
        }

        /**
         * Answers whether this module is available on the current system.
         * This implementation always returns 'true'.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        virtual bool IsAvailable(void) const {
            return C::IsAvailable();
        }

        /**
         * Answers whether this description is describing the class of
         * 'module'.
         *
         * @param module The module to test.
         *
         * @return 'true' if 'module' is described by this description,
         *         'false' otherwise.
         */
        virtual bool IsDescribing(const Module * module) const {
            //return dynamic_cast<const C*>(module) != NULL;
            // mueller: The version above depends on modules being ordered from
            // most-specialised to base classes. This is cannot be guaranteed
            // and has the effect that the network serialisation uses wrong
            // types. The new RTTI version only allows exact matches. We believe
            // that this does not have any negative effects somewhere else in the
            // programme.
            return (typeid(C) == typeid(*module));
        }

        /**
         * Answer whether or not this module can be used in a quickstart
         *
         * @return 'true' if the module can be used in a quickstart
         */
        virtual bool IsVisibleForQuickstart(void) const {
            return C::SupportQuickstart();
        }

    protected:

        /**
         * Creates a new module object from this description.
         *
         * @return The newly created module object or 'NULL' in case of an
         *         error.
         */
        virtual Module *createModuleImpl(void) const {
            try {
                Module *m = new C();
                return m;
            } catch(vislib::Exception ex) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Exception while creating module %s: %s\n",
                    C::ClassName(), ex.GetMsgA());
            } catch(...) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Exception while creating module %s: Unknown exception\n",
                    C::ClassName());
            }
            return NULL;
        }

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MODULEAUTODESCRIPTION_H_INCLUDED */
