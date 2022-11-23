/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "BlockVolumeMesh.h"
#include "CoordSysMarker.h"
#include "ModernTrisoupRenderer.h"
#include "TriSoupDataSource.h"
#include "TriSoupRenderer.h"
#include "WavefrontObjDataSource.h"
#include "vislib/Trace.h"
#include "volumetrics/IsoSurface.h"
#include "volumetrics/VoluMetricJob.h"

namespace megamol::trisoup_gl {
class TrisoupGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(TrisoupGLPluginInstance)

public:
    TrisoupGLPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("trisoup_gl", "Plugin for rendering TriSoup mesh data") {
        vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
    };

    ~TrisoupGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::TriSoupRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::ModernTrisoupRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::TriSoupDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::WavefrontObjDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::BlockVolumeMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::volumetrics::VoluMetricJob>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::CoordSysMarker>();
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::volumetrics::IsoSurface>();
    }
};
} // namespace megamol::trisoup_gl
