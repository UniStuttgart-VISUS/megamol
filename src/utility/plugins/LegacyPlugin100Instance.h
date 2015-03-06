/*
 * LegacyPlugin100Instance.h
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_LEGACYPLUGIN100INSTANCE_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_LEGACYPLUGIN100INSTANCE_H_INCLUDED
#pragma once

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "utility/plugins/PluginManager.h"
#include <memory>
#include "vislib/tchar.h"
#include <string>
#include "vislib/sys/DynamicLinkLibrary.h"


namespace megamol {
namespace core {
namespace utility {
namespace plugins {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class LegacyPlugin100Instance : public AbstractPluginInstance {
    public:

        /**
         * Continue to load 'lib' as plugin of version 1.00
         *
         * @param path The plugin's file system path
         * @param lib The loaded library
         * @param coreInst The calling core instance
         *
         * @return A collection containing the loaded plugin
         *
         * @throw vislib::Exception on failure
         */
        static PluginManager::collection_type ContinueLoad(
            const std::basic_string<TCHAR> &path,
            std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib,
            ::megamol::core::CoreInstance& coreInst);

        /** Dtor. */
        virtual ~LegacyPlugin100Instance(void);

    private:

        /** Ctor. */
        LegacyPlugin100Instance(const char *asm_name, const char *description,
            std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib);

        /** deleted copy ctor */
        LegacyPlugin100Instance(const LegacyPlugin100Instance& src) = delete;

        /** deleted assignment operatior */
        LegacyPlugin100Instance& operator=(const LegacyPlugin100Instance& rhs) = delete;

        /** The plugin library object */
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib;

    };

} /* end namespace plugins */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_LEGACYPLUGIN100INSTANCE_H_INCLUDED */
