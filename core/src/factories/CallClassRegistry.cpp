/*
 * CallClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "factories/CallClassRegistry.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"

#include "mmcore/DataWriterCtrlCall.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/EventCall.h"
#include "mmcore/cluster/CallRegisterAtController.h"
#include "mmcore/cluster/SyncDataSourcesCall.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/flags/FlagCalls.h"
#include "mmcore/job/TickCall.h"
#include "mmcore/param/ParamCalls.h"
#include "mmcore/view/Call6DofInteraction.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallTimeControl.h"
#include "mmcore/view/light/CallLight.h"

using namespace megamol::core;


/*
 * factories::register_call_classes
 */
void factories::register_call_classes(factories::CallDescriptionManager& instance) {
    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph call descriptions here
    //////////////////////////////////////////////////////////////////////
    instance.RegisterAutoDescription<cluster::CallRegisterAtController>();
    instance.RegisterAutoDescription<view::CallClipPlane>();
    instance.RegisterAutoDescription<view::CallTimeControl>();
    instance.RegisterAutoDescription<DataWriterCtrlCall>();
    instance.RegisterAutoDescription<view::Call6DofInteraction>();
    instance.RegisterAutoDescription<cluster::mpi::MpiCall>();
    instance.RegisterAutoDescription<view::light::CallLight>();
    instance.RegisterAutoDescription<job::TickCall>();
    instance.RegisterAutoDescription<DirectDataWriterCall>();
    instance.RegisterAutoDescription<cluster::SyncDataSourcesCall>();
    instance.RegisterAutoDescription<FlagCallRead_CPU>();
    instance.RegisterAutoDescription<FlagCallWrite_CPU>();
    instance.RegisterAutoDescription<CallEvent>();
    instance.RegisterAutoDescription<view::CallRender3D>();
    instance.RegisterAutoDescription<param::FloatParamCall>();
    instance.RegisterAutoDescription<param::IntParamCall>();
    instance.RegisterAutoDescription<view::CallGetTransferFunction>();
}
