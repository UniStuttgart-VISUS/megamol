/*
 * RootModuleNamespace.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED
#define MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "ModuleNamespace.h"
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"


namespace megamol {
namespace core {


    /**
     * Class represents the root namespace for the module namespace
     */
    class MEGAMOLCORE_API RootModuleNamespace: public ModuleNamespace {
    public:

        /**
         * Ctor.
         */
        RootModuleNamespace(void);

        /**
         * Dtor.
         */
        virtual ~RootModuleNamespace(void);

        /**
         * Answer the full namespace path for 'path' seen relative to 'base'.
         *
         * @param base The base path
         * @param path The path
         *
         * @return The full namespace path
         */
        vislib::StringA FullNamespace(const vislib::StringA& base,
            const vislib::StringA& path) const;

        /**
         * Finds the specified module namespace
         *
         * @param path A array with the separated namespace names.
         * @param createMissing If true, all missing namespace object will be created.
         *
         * @return The requested namespace or 'NULL' in case of an error.
         */
        ModuleNamespace * FindNamespace(const vislib::Array<vislib::StringA>& path, bool createMissing);

        /**
         * Locks the module namespace
         *
         * @param write If 'true' locks the namespace for writing, if 'false' locks only for reading.
         */
        virtual void LockModuleGraph(bool write);

        /**
         * Unlocks the module namespace
         */
        virtual void UnlockModuleGraph(void);

    protected:

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The graph access synchronization object */
        vislib::sys::CriticalSection lock;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ROOTMODULENAMESPACE_H_INCLUDED */
