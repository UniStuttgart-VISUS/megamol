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
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/ParticleGridDataCall.h"
#include "mmcore/cluster/CallRegisterAtController.h"
#include "mmcore/cluster/simple/ClientViewRegistration.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/CallTimeControl.h"
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
#include "mmcore/view/CallUpdateDirect3D.h"
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
#include "mmcore/view/CallCamParams.h"
#include "mmcore/view/CallCamParamSync.h"
#include "mmcore/view/CallSplitViewOverlay.h"
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

using namespace megamol::core;


/*
 * factories::register_call_classes
 */
void factories::register_call_classes(factories::CallDescriptionManager& instance) {
    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph call descriptions here
    //////////////////////////////////////////////////////////////////////
    instance.RegisterAutoDescription<cluster::CallRegisterAtController>();
    instance.RegisterAutoDescription<cluster::simple::ClientViewRegistration>();
    instance.RegisterAutoDescription<moldyn::DirectionalParticleDataCall>();
    instance.RegisterAutoDescription<moldyn::MultiParticleDataCall>();
    instance.RegisterAutoDescription<moldyn::ParticleGridDataCall>();
    instance.RegisterAutoDescription<view::CallClipPlane>();
    instance.RegisterAutoDescription<view::CallGetTransferFunction>();
    instance.RegisterAutoDescription<view::CallRender2D>();
    instance.RegisterAutoDescription<view::CallRender3D>();
    instance.RegisterAutoDescription<view::CallRenderDeferred3D>();
    instance.RegisterAutoDescription<view::CallRenderView>();
    instance.RegisterAutoDescription<view::CallTimeControl>();
    instance.RegisterAutoDescription<view::CallSplitViewOverlay>();
    instance.RegisterAutoDescription<DataWriterCtrlCall>();
    instance.RegisterAutoDescription<CallVolumeData>();
    instance.RegisterAutoDescription<moldyn::VolumeDataCall>();
    instance.RegisterAutoDescription<misc::BezierCurvesListDataCall>();
    instance.RegisterAutoDescription<misc::VolumetricDataCall>();
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    instance.RegisterAutoDescription<view::CallUpdateDirect3D>();
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
    instance.RegisterAutoDescription<view::CallCamParams>();
    instance.RegisterAutoDescription<view::CallCamParamSync>();
    instance.RegisterAutoDescription<misc::QRCodeDataCall>();
    instance.RegisterAutoDescription<misc::CalloutImageCall>();
    instance.RegisterAutoDescription<view::Call6DofInteraction>();
    instance.RegisterAutoDescription<cluster::mpi::MpiCall>();
    instance.RegisterAutoDescription<moldyn::EllipsoidalParticleDataCall>();
    instance.RegisterAutoDescription<moldyn::ParticleRelistCall>();
}
