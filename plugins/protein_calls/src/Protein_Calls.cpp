/*
 * Protein_Calls.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/CrystalStructureDataCall.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/IntSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/SplitMergeCall.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/VariantMatchDataCall.h"
#include "protein_calls/CallMouseInput.h"
#include "protein_calls/PerAtomFloatCall.h"
#include "protein_calls/TunnelResidueDataCall.h"

namespace megamol::protein_calls {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "Protein_Calls", // TODO: Change this!

                /* human-readable plugin description */
                "Plugin containing calls used by Protein and Protein_CUDA") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::Protein_Calls::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::Protein_Calls::MyModule2>();
            //   ...
            //

            // register calls here:
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::BindingSiteCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::CrystalStructureDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::DiagramCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::IntSelectionCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::MolecularDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::ResidueSelectionCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::SplitMergeCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::VariantMatchDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::VTIDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::CallMouseInput>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::PerAtomFloatCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::TunnelResidueDataCall>();
        }
    };
} // namespace megamol::protein_calls
