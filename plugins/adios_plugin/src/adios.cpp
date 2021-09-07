/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ADIOSFlexConvert.h"
#include "ADIOStoMultiParticle.h"
#include "ADIOStoTable.h"
#include "MultiParticletoADIOS.h"
#include "TableToADIOS.h"
#include "adiosDataSource.h"
#include "adiosWriter.h"
#include "adios_plugin/CallADIOSData.h"
#include "ls1ParticleFormat.h"

namespace megamol::adios {
    class AdiosPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(AdiosPluginInstance)

    public:
        AdiosPluginInstance() : megamol::core::utility::plugins::AbstractPluginInstance("adios", "The adios plugin."){};

        ~AdiosPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::adios::adiosDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOStoMultiParticle>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::adiosWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::MultiParticletoADIOS>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::TableToADIOS>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOSFlexConvert>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ADIOStoTable>();
            this->module_descriptions.RegisterAutoDescription<megamol::adios::ls1ParticleFormat>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::adios::CallADIOSData>();
        }
    };
} // namespace megamol::adios
