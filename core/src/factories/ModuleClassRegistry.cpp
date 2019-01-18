/*
 * ModuleClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "factories/ModuleClassRegistry.h"

#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/LoaderADModuleAutoDescription.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/ModuleDescription.h"

#include "mmcore/cluster/ClusterController.h"
#include "mmcore/cluster/ClusterViewMaster.h"
#include "mmcore/cluster/PowerwallView.h"
#include "mmcore/cluster/mpi/MpiProvider.h"
#include "mmcore/cluster/mpi/View.h"
#include "mmcore/cluster/simple/Client.h"
#include "mmcore/cluster/simple/Heartbeat.h"
#include "mmcore/cluster/simple/Server.h"
#include "mmcore/cluster/simple/View.h"
#include "mmcore/misc/SiffCSplineFitter.h"
#include "mmcore/misc/WatermarkRenderer.h"
#include "mmcore/misc/TestSpheresDataSource.h"
#include "mmcore/moldyn/AddParticleColours.h"
#include "mmcore/moldyn/ArrowRenderer.h"
#include "mmcore/moldyn/DataGridder.h"
#include "mmcore/moldyn/GrimRenderer.h"
#include "mmcore/moldyn/MipDepthSphereRenderer.h"
#include "mmcore/moldyn/MMPGDDataSource.h"
#include "mmcore/moldyn/MMPGDWriter.h"
#include "mmcore/moldyn/MMPLDDataSource.h"
#include "mmcore/moldyn/MMPLDWriter.h"
#include "mmcore/moldyn/OracleSphereRenderer.h"
#include "mmcore/moldyn/SimpleSphereRenderer.h"
#include "mmcore/moldyn/SphereOutlineRenderer.h"
#include "mmcore/moldyn/DirPartColModulate.h"
#include "mmcore/moldyn/DirPartFilter.h"
#include "mmcore/moldyn/ParticleListFilter.h"
//#include "mmcore/special/ColStereoDisplay.h"
#include "mmcore/special/StubModule.h"
#include "mmcore/view/ClipPlane.h"
#include "mmcore/view/LinearTransferFunction.h"
#include "mmcore/view/TransferFunctionRenderer.h"
#include "mmcore/view/MuxRenderer3D.h"
#include "mmcore/view/special/AnaglyphStereoView.h"
#include "mmcore/view/special/ChronoGraph.h"
#include "mmcore/view/special/DemoRenderer2D.h"
#include "mmcore/view/special/QuadBufferStereoView.h"
#include "mmcore/view/special/ScreenShooter.h"
#include "mmcore/view/SwitchRenderer3D.h"
#include "mmcore/view/TileView.h"
#include "mmcore/view/View2D.h"
#include "mmcore/view/View3D.h"
#include "mmcore/view/RendererRegistration.h"
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
#include "mmcore/view/ViewDirect3D.h"
#include "mmcore/moldyn/D3D11SimpleSphereRenderer.h"
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
#include "mmcore/view/BlinnPhongRendererDeferred.h"
#include "mmcore/view/SplitView.h"
#include "mmcore/view/SharedCameraParameters.h"
#include "mmcore/view/LinkedView3D.h"
#include "mmcore/job/DataWriterJob.h"
#include "mmcore/job/JobThread.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "mmcore/moldyn/AddClusterColours.h"
#include "mmcore/moldyn/DynDensityGradientEstimator.h"
#include "job/PluginsStateFileGeneratorJob.h"
#include "mmcore/utility/LuaHostSettingsModule.h"

using namespace megamol::core;


/*
 * factories::register_module_classes
 */
void factories::register_module_classes(factories::ModuleDescriptionManager& instance) {

    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph module descriptions here
    //////////////////////////////////////////////////////////////////////

    instance.RegisterAutoDescription<cluster::ClusterController>();
    instance.RegisterAutoDescription<cluster::ClusterViewMaster>();
    instance.RegisterAutoDescription<cluster::PowerwallView>();
    instance.RegisterAutoDescription<cluster::simple::Client>();
    instance.RegisterAutoDescription<cluster::simple::Heartbeat>();
    instance.RegisterAutoDescription<cluster::simple::Server>();
    instance.RegisterAutoDescription<cluster::simple::View>();
    instance.RegisterAutoDescription<cluster::mpi::MpiProvider>();
    instance.RegisterAutoDescription<cluster::mpi::View>();
    instance.RegisterAutoDescription<misc::SiffCSplineFitter>();
    instance.RegisterAutoDescription<misc::WatermarkRenderer>();
    instance.RegisterAutoDescription<misc::TestSpheresDataSource>();
    instance.RegisterAutoDescription<moldyn::AddParticleColours>();
    instance.RegisterAutoDescription<moldyn::ArrowRenderer>();
    instance.RegisterAutoDescription<moldyn::DataGridder>();
    instance.RegisterAutoDescription<moldyn::GrimRenderer>();
    instance.RegisterAutoDescription<moldyn::MipDepthSphereRenderer>();
    instance.RegisterAutoDescription<moldyn::MMPGDDataSource>();
    instance.RegisterAutoDescription<moldyn::MMPGDWriter>();
    instance.RegisterAutoDescription<moldyn::MMPLDDataSource>();
    instance.RegisterAutoDescription<moldyn::MMPLDWriter>();
    instance.RegisterAutoDescription<moldyn::SimpleSphereRenderer>();
    instance.RegisterAutoDescription<moldyn::SphereOutlineRenderer>();
    instance.RegisterAutoDescription<moldyn::OracleSphereRenderer>();
    instance.RegisterAutoDescription<moldyn::DirPartColModulate>();
    instance.RegisterAutoDescription<moldyn::ParticleListFilter>();
    instance.RegisterAutoDescription<moldyn::DirPartFilter>();
    //instance.RegisterAutoDescription<special::ColStereoDisplay>();
    instance.RegisterAutoDescription<special::StubModule>();
    instance.RegisterAutoDescription<view::ClipPlane>();
    instance.RegisterAutoDescription<view::LinearTransferFunction>();
    instance.RegisterAutoDescription<view::TransferFunctionRenderer>();
    instance.RegisterAutoDescription<view::MuxRenderer3D<2> >();
    instance.RegisterAutoDescription<view::MuxRenderer3D<3> >();
    instance.RegisterAutoDescription<view::MuxRenderer3D<4> >();
    instance.RegisterAutoDescription<view::MuxRenderer3D<5> >();
    instance.RegisterAutoDescription<view::MuxRenderer3D<10> >();
    instance.RegisterAutoDescription<view::special::AnaglyphStereoView>();
    instance.RegisterAutoDescription<view::special::ChronoGraph>();
    instance.RegisterAutoDescription<view::special::DemoRenderer2D>();
    instance.RegisterAutoDescription<view::special::QuadBufferStereoView>();
    instance.RegisterAutoDescription<view::special::ScreenShooter>();
    instance.RegisterAutoDescription<view::SwitchRenderer3D>();
    instance.RegisterAutoDescription<view::TileView>();
    instance.RegisterAutoDescription<view::View2D>();
    instance.RegisterAutoDescription<view::View3D>();
    instance.RegisterAutoDescription<view::BlinnPhongRendererDeferred>();
    instance.RegisterAutoDescription<view::SplitView>();
    instance.RegisterAutoDescription<view::SharedCameraParameters>();
    instance.RegisterAutoDescription<view::LinkedView3D>();
    instance.RegisterAutoDescription<view::RendererRegistration>();
    instance.RegisterAutoDescription<job::DataWriterJob>();
    instance.RegisterAutoDescription<job::JobThread>();
    instance.RegisterAutoDescription<moldyn::AddClusterColours>();
    instance.RegisterAutoDescription<moldyn::DynDensityGradientEstimator>();
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    instance.RegisterAutoDescription<view::ViewDirect3D>();
    instance.RegisterAutoDescription<moldyn::D3D11SimpleSphereRenderer>();
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
    instance.RegisterAutoDescription<job::PluginsStateFileGeneratorJob>();
    instance.RegisterAutoDescription<core::utility::LuaHostSettingsModule>();
}
