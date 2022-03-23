/*
 * ModuleClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "factories/ModuleClassRegistry.h"
#include "stdafx.h"

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"

#include "job/TickSwitch.h"
#include "mmcore/EventStorage.h"
#include "mmcore/FileStreamProvider.h"
#include "mmcore/ResourceTestModule.h"
#include "mmcore/cluster/ClusterController.h"
#include "mmcore/cluster/mpi/MpiProvider.h"
#include "mmcore/flags/FlagStorage.h"
#include "mmcore/job/DataWriterJob.h"
#include "mmcore/job/JobThread.h"
#include "mmcore/param/GenericParamModule.h"
#include "mmcore/special/StubModule.h"
#include "mmcore/utility/LuaHostSettingsModule.h"
#include "mmcore/view/ClipPlane.h"
#include "mmcore/view/TransferFunction.h"
#include "mmcore/view/View3D.h"
#include "mmcore/view/light/AmbientLight.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/HDRILight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore/view/light/QuadLight.h"
#include "mmcore/view/light/SpotLight.h"
#include "mmcore/view/light/TriDirectionalLighting.h"


using namespace megamol::core;


/*
 * factories::register_module_classes
 */
void factories::register_module_classes(factories::ModuleDescriptionManager& instance) {

    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph module descriptions here
    //////////////////////////////////////////////////////////////////////

    instance.RegisterAutoDescription<cluster::ClusterController>();
    instance.RegisterAutoDescription<cluster::mpi::MpiProvider>();
    instance.RegisterAutoDescription<special::StubModule>();
    instance.RegisterAutoDescription<view::ClipPlane>();
    instance.RegisterAutoDescription<job::DataWriterJob>();
    instance.RegisterAutoDescription<job::JobThread>();
    instance.RegisterAutoDescription<core::utility::LuaHostSettingsModule>();
    instance.RegisterAutoDescription<core::job::TickSwitch>();
    instance.RegisterAutoDescription<core::FileStreamProvider>();
    instance.RegisterAutoDescription<view::light::AmbientLight>();
    instance.RegisterAutoDescription<view::light::DistantLight>();
    instance.RegisterAutoDescription<view::light::HDRILight>();
    instance.RegisterAutoDescription<view::light::PointLight>();
    instance.RegisterAutoDescription<view::light::QuadLight>();
    instance.RegisterAutoDescription<view::light::SpotLight>();
    instance.RegisterAutoDescription<view::light::TriDirectionalLighting>();
    instance.RegisterAutoDescription<FlagStorage>();
    instance.RegisterAutoDescription<EventStorage>();
    instance.RegisterAutoDescription<view::View3D>();
    instance.RegisterAutoDescription<ResourceTestModule>();
    instance.RegisterAutoDescription<param::FloatParamModule>();
    instance.RegisterAutoDescription<param::IntParamModule>();
    instance.RegisterAutoDescription<view::TransferFunction>();
}
