/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "geometry_calls/BezierCurvesListDataCall.h"
#include "geometry_calls/CalloutImageCall.h"
#include "geometry_calls/EllipsoidalDataCall.h"
#include "geometry_calls/LinesDataCall.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "geometry_calls/ParticleRelistCall.h"
#include "geometry_calls/QRCodeDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"

namespace megamol::geocalls {
class GeometryCallsPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(GeometryCallsPluginInstance)

public:
    GeometryCallsPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("geometry_calls", "The geometry_calls plugin."){};

    ~GeometryCallsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::LinesDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::MultiParticleDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::EllipsoidalParticleDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::ParticleRelistCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::VolumetricDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::BezierCurvesListDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::QRCodeDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls::CalloutImageCall>();
    }
};
} // namespace megamol::geocalls
