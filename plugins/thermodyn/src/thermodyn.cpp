/*
 * thermodyn.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "PhaseSeparator.h"
#include "PhaseAnimator.h"
#include "ParticleSurface.h"
#include "VelocityDistribution.h"
#include "ParticleSpawner.h"
#include "rendering/BoxRenderer.h"
#include "rendering/TimeLinePlot.h"
#include "PointMeshDistance.h"
#include "PointInterfaceClassification.h"
#include "AccumulateInterfacePresence.h"
#include "PrepareSurfaceEvents.h"
#include "PointSurfaceElementsDistance.h"
#include "AccumulateInterfacePresence2.h"
#include "PrepareSurfaceEvents2.h"
#include "SphereWidget.h"
#include "MeshWidget.h"
#include "ParticlePaths.h"
#include "PathDump.h"
#include "PathReader.h"
#include "PathSelection.h"
#include "IDBroker.h"
#include "ParticleSurface2.h"
#include "MeshAddColor.h"
#include "MeshExtrude.h"
#include "ParticlesInsideMesh.h"
#include "ParticleMeshTracking.h"
#include "ParticleSurfaceRefinement.h"

#include "thermodyn/BoxDataCall.h"
#include "thermodyn/CallStatsInfo.h"
#include "thermodyn/CallEvents.h"

namespace megamol::thermodyn {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "thermodyn", // TODO: Change this!

              /* human-readable plugin description */
              "Describing thermodyn (TODO: Change this!)"){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseSeparator>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseAnimator>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticleSurface>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::VelocityDistribution>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticleSpawner>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::rendering::BoxRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::rendering::TimeLinePlot>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PointMeshDistance>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PointInterfaceClassification>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::AccumulateInterfacePresence>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PrepareSurfaceEvents>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PointSurfaceElementsDistance>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::AccumulateInterfacePresence2>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PrepareSurfaceEvents2>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::SphereWidget>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MeshWidget>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticlePaths>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PathDump>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PathReader>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PathSelection>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::IDBroker>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticleSurface2>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MeshAddColor>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MeshExtrude>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticlesInsideMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticleMeshTracking>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::ParticleSurfaceRefinement>();
        //
        // TODO: Register your plugin's modules here
        // like:
        //   this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MyModule2>();
        //   ...
        //

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::BoxDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::CallStatsInfo>();
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::CallEvents>();
        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::MyCall2>();
        //   ...
        //
    }
};
} // namespace megamol::thermodyn
