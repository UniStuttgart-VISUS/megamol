/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/CrystalStructureDataCall.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/IntSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/PerAtomFloatCall.h"
#include "protein_calls/RamachandranDataCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/SplitMergeCall.h"
#include "protein_calls/TunnelResidueDataCall.h"
#include "protein_calls/UncertaintyDataCall.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/VariantMatchDataCall.h"

namespace megamol::protein_calls {
class ProteinCallsPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinCallsPluginInstance)

public:
    ProteinCallsPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "Protein_Calls", "Plugin containing calls used by Protein and Protein_CUDA"){};

    ~ProteinCallsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::BindingSiteCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::CrystalStructureDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::DiagramCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::IntSelectionCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::MolecularDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::ResidueSelectionCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::SplitMergeCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::VariantMatchDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::VTIDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::PerAtomFloatCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::TunnelResidueDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::RamachandranDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::UncertaintyDataCall>();
    }
};
} // namespace megamol::protein_calls
