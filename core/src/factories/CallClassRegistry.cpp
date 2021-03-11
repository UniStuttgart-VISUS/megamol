/*
 * CallClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "factories/CallClassRegistry.h"

#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/CallDescription.h"

#include "mmcore/DataWriterCtrlCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/cluster/CallRegisterAtController.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/CallRender3DGL.h"
#include "mmcore/view/CallRenderViewGL.h"
#include "mmcore/view/CallTimeControl.h"
#include "mmcore/CallVolumeData.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/misc/QRCodeDataCall.h"
#include "mmcore/misc/CalloutImageCall.h"
#include "mmcore/view/Call6DofInteraction.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/moldyn/EllipsoidalDataCall.h"
#include "mmcore/moldyn/ParticleRelistCall.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore/job/TickCall.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/cluster/SyncDataSourcesCall.h"
#include "mmcore/view/special/CallbackScreenShooter.h"
#include "mmcore/FlagCall.h"
#include "mmcore/FlagCall_GL.h"
#include "mmcore/view/CallRender3D.h"

using namespace megamol::core;


/*
 * factories::register_call_classes
 */
void factories::register_call_classes(factories::CallDescriptionManager& instance) {
    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph call descriptions here
    //////////////////////////////////////////////////////////////////////
    instance.RegisterAutoDescription<cluster::CallRegisterAtController>();
    instance.RegisterAutoDescription<moldyn::MultiParticleDataCall>();
    instance.RegisterAutoDescription<view::CallClipPlane>();
    instance.RegisterAutoDescription<view::CallGetTransferFunction>();
    instance.RegisterAutoDescription<view::CallRender2DGL>();
    instance.RegisterAutoDescription<view::CallRender3DGL>();
    instance.RegisterAutoDescription<view::CallRenderViewGL>();
    instance.RegisterAutoDescription<view::CallTimeControl>();
    instance.RegisterAutoDescription<DataWriterCtrlCall>();
    instance.RegisterAutoDescription<CallVolumeData>();
    instance.RegisterAutoDescription<moldyn::VolumeDataCall>();
    instance.RegisterAutoDescription<misc::BezierCurvesListDataCall>();
    instance.RegisterAutoDescription<misc::VolumetricDataCall>();
    instance.RegisterAutoDescription<misc::QRCodeDataCall>();
    instance.RegisterAutoDescription<misc::CalloutImageCall>();
    instance.RegisterAutoDescription<view::Call6DofInteraction>();
    instance.RegisterAutoDescription<cluster::mpi::MpiCall>();
    instance.RegisterAutoDescription<moldyn::EllipsoidalParticleDataCall>();
    instance.RegisterAutoDescription<moldyn::ParticleRelistCall>();
    instance.RegisterAutoDescription<view::light::CallLight>();
    instance.RegisterAutoDescription<job::TickCall>();
    instance.RegisterAutoDescription<DirectDataWriterCall>();
    instance.RegisterAutoDescription<cluster::SyncDataSourcesCall>();
    instance.RegisterAutoDescription<view::special::CallbackScreenShooterCall>();
    instance.RegisterAutoDescription<FlagCall>();
    instance.RegisterAutoDescription<FlagCallRead_GL>();
    instance.RegisterAutoDescription<FlagCallWrite_GL>();
    instance.RegisterAutoDescription<view::CallRender3D>();
}
