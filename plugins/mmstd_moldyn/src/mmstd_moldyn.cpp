/*
 * mmstd.moldyn.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/LoaderADModuleAutoDescription.h"

#include "mmstd_moldyn/BrickStatsCall.h"

#include "io/IMDAtomDataSource.h"
#include "io/MMSPDDataSource.h"
#include "io/SIFFDataSource.h"
#include "io/SIFFWriter.h"
#include "io/VIMDataSource.h"
#include "io/VisIttDataSource.h"
#include "io/VTFDataSource.h"
#include "io/VTFResDataSource.h"
#include "io/XYZLoader.h"
#include "io/TclMolSelectionLoader.h"
#include "io/BrickStatsDataSource.h"
#include "io/MMPGDDataSource.h"
#include "io/MMPGDWriter.h"

#include "misc/ParticleWorker.h"

#include "rendering/DataGridder.h"
#include "rendering/GrimRenderer.h"
#include "rendering/SphereRenderer.h"
#include "rendering/ArrowRenderer.h"
#include "rendering/ParticleGridDataCall.h"

namespace megamol::stdplugin::moldyn {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : megamol::core::utility::plugins::Plugin200Instance(
                /* machine-readable plugin assembly name */
                "mmstd_moldyn",
                /* human-readable plugin description */
                "MegaMol Plugins for Molecular Dynamics Data Visualization") {
            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {
            // register modules here:
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::IMDAtomDataSource> >();
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::MMSPDDataSource> >();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::SIFFDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::SIFFWriter>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VIMDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VisIttDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VTFDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VTFResDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::misc::ParticleWorker>();
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::XYZLoader> >();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::TclMolSelectionLoader>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::BrickStatsDataSource>();
			this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::DataGridder>();
			this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::GrimRenderer>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::ArrowRenderer>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::SphereRenderer>();
			this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::MMPGDDataSource>();
			this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::MMPGDWriter>();
            // register calls here:
            this->call_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::BrickStatsCall>();
			this->call_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::ParticleGridDataCall>();
        }
    };
} // namespace megamol::stdplugin::moldyn
