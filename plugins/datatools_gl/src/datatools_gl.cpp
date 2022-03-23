/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "MeshTranslateRotateScale.h"
#include "io/PLYDataSource.h"
#include "io/PlyWriter.h"
#include "io/STLDataSource.h"
#include "io/TriMeshSTLWriter.h"
#include "misc/AddClusterColours.h"
#include "misc/ParticleDensityOpacityModule.h"
#include "misc/ParticleInspector.h"
#include "misc/ParticleListMergeModule.h"
#include "misc/ParticleWorker.h"


namespace megamol::datatools_gl {
class DatatoolsGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(DatatoolsGLPluginInstance)

public:
    DatatoolsGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "datatools_gl", "MegaMol Standard-Plugin containing data manipulation and conversion modules"){};

    ~DatatoolsGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::misc::ParticleDensityOpacityModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::misc::ParticleInspector>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::misc::ParticleListMergeModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::misc::ParticleWorker>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::misc::AddClusterColours>();

        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::io::PlyWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::io::STLDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::io::TriMeshSTLWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::io::PLYDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools_gl::MeshTranslateRotateScale>();
    }
};
} // namespace megamol::datatools_gl
