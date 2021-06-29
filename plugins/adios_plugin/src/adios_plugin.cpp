/*
 * adios_plugin.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "adiosDataSource.h"
#include "adios_plugin/CallADIOSData.h"
#include "ADIOStoMultiParticle.h"
#include "MultiParticletoADIOS.h"
#include "adiosWriter.h"
#include "TableToADIOS.h"
#include "ADIOSFlexConvert.h"
#include "ADIOStoTable.h"
#include "ls1ParticleFormat.h"

namespace megamol::adios {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "adios_plugin", // TODO: Change this!

                /* human-readable plugin description */
                "Describing adios_plugin (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            this->module_descriptions.RegisterAutoDescription<megamol::adios::adiosDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOStoMultiParticle>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::adiosWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::MultiParticletoADIOS>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::TableToADIOS>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOSFlexConvert>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOStoTable>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ls1ParticleFormat>();

            // register calls here:

            this->call_descriptions.RegisterAutoDescription<megamol::adios::CallADIOSData>();


        }
    };
} // namespace megamol::adios
