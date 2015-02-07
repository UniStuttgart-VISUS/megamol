/*
 * ModuleDescriptionManager.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
//#include "mmcore/factories/LoaderADModuleAutoDescription.h"
//#include "mmcore/factories/ModuleAutoDescription.h"
//#include "mmcore/factories/ModuleDescription.h"
//#include "vislib/assert.h"


/*
 * megamol::core::factories::ModuleDescriptionManager::ModuleDescriptionManager
 */
megamol::core::factories::ModuleDescriptionManager::ModuleDescriptionManager()
        : ObjectDescriptionManager<megamol::core::factories::ModuleDescription>() {
    // intentionally empty
}


/*
 * megamol::core::factories::ModuleDescriptionManager::~ModuleDescriptionManager
 */
megamol::core::factories::ModuleDescriptionManager::~ModuleDescriptionManager() {
    // intentionally empty
}


//#include "mmcore/cluster/ClusterController.h"
//#include "mmcore/cluster/ClusterViewMaster.h"
//#include "mmcore/cluster/PowerwallView.h"
//#include "mmcore/cluster/mpi/MpiProvider.h"
//#include "mmcore/cluster/mpi/View.h"
//#include "mmcore/cluster/simple/Client.h"
//#include "mmcore/cluster/simple/Heartbeat.h"
//#include "mmcore/cluster/simple/Server.h"
//#include "mmcore/cluster/simple/View.h"
//#include "mmcore/misc/ImageViewer.h"
//#include "mmcore/misc/LinesRenderer.h"
//#include "mmcore/misc/SiffCSplineFitter.h"
//#include "mmcore/misc/TestSpheresDataSource.h"
//#include "mmcore/moldyn/AddParticleColours.h"
//#include "mmcore/moldyn/ArrowRenderer.h"
//#include "mmcore/moldyn/DataGridder.h"
//#include "mmcore/moldyn/GrimRenderer.h"
//#include "mmcore/moldyn/IMDAtomDataSource.h"
//#include "mmcore/moldyn/MipDepthSphereRenderer.h"
//#include "mmcore/moldyn/MMPGDDataSource.h"
//#include "mmcore/moldyn/MMPGDWriter.h"
//#include "mmcore/moldyn/MMPLDDataSource.h"
//#include "mmcore/moldyn/MMPLDWriter.h"
//#include "mmcore/moldyn/MMSPDDataSource.h"
//#include "mmcore/moldyn/OracleSphereRenderer.h"
//#include "mmcore/moldyn/SIFFDataSource.h"
//#include "mmcore/moldyn/SimpleGeoSphereRenderer.h"
//#include "mmcore/moldyn/SimpleSphereRenderer.h"
//#include "mmcore/moldyn/NGSphereRenderer.h"
//#include "mmcore/moldyn/ClusteredSphereRenderer.h"
//#include "mmcore/moldyn/SphereOutlineRenderer.h"
//#include "mmcore/moldyn/VIMDataSource.h"
//#include "mmcore/moldyn/VisIttDataSource.h"
//#include "mmcore/moldyn/DirPartColModulate.h"
//#include "mmcore/moldyn/DirPartFilter.h"
//#include "mmcore/moldyn/ParticleListFilter.h"
//#include "mmcore/moldyn/ParticleWorker.h"
//#include "mmcore/moldyn/D3D11SimpleSphereRenderer.h"
////#include "mmcore/special/ColStereoDisplay.h"
//#include "mmcore/view/ClipPlane.h"
//#include "mmcore/view/LinearTransferFunction.h"
//#include "mmcore/view/TransferFunctionRenderer.h"
//#include "mmcore/view/MuxRenderer3D.h"
//#include "mmcore/view/special/AnaglyphStereoView.h"
//#include "mmcore/view/special/ChronoGraph.h"
//#include "mmcore/view/special/DemoRenderer2D.h"
//#include "mmcore/view/special/QuadBufferStereoView.h"
//#include "mmcore/view/special/ScreenShooter.h"
//#include "mmcore/view/SwitchRenderer3D.h"
//#include "mmcore/view/TileView.h"
//#include "mmcore/view/View2D.h"
//#include "mmcore/view/View3D.h"
//#include "mmcore/view/ViewDirect3D.h"
//#include "mmcore/view/BlinnPhongRendererDeferred.h"
//#include "mmcore/view/SplitView.h"
//#include "mmcore/view/SharedCameraParameters.h"
//#include "mmcore/view/LinkedView3D.h"
//#include "mmcore/job/DataWriterJob.h"
//#include "mmcore/job/JobThread.h"
////#include "mmcore/vismol2/Mol20DataSource.h"
////#include "mmcore/vismol2/Mol20Renderer.h"
//#include "mmcore/moldyn/VolumeDataCall.h"
//#include "mmcore/moldyn/SIFFWriter.h"
//#include "mmcore/moldyn/VTFDataSource.h"
//#include "mmcore/moldyn/VTFResDataSource.h"
//#include "mmcore/moldyn/AddClusterColours.h"
//#include "mmcore/moldyn/DynDensityGradientEstimator.h"
//
//using namespace megamol::core;
//
//
///*
// * ModuleDescriptionManager::inst
// */
//vislib::SmartPtr<ModuleDescriptionManager> ModuleDescriptionManager::inst;
//
//
///*
// * ModuleDescriptionManager::Instance
// */
//ModuleDescriptionManager * ModuleDescriptionManager::Instance() {
//    if (inst.IsNull()) {
//        inst = new ModuleDescriptionManager();
//        registerObjects(inst.operator->());
//    }
//    return inst.operator->();
//}
//
//
///*
// * ModuleDescriptionManager::ShutdownInstance
// */
//void ModuleDescriptionManager::ShutdownInstance() {
//    inst = NULL;
//}
//
//
///*
// * ModuleDescriptionManager::registerObjects
// */
//void ModuleDescriptionManager::registerObjects(ModuleDescriptionManager *instance) {
//
//    //////////////////////////////////////////////////////////////////////
//    // Register all rendering graph module descriptions here
//    //////////////////////////////////////////////////////////////////////
//
//    instance->registerAutoDescription<cluster::ClusterController>();
//    instance->registerAutoDescription<cluster::ClusterViewMaster>();
//    instance->registerAutoDescription<cluster::PowerwallView>();
//    instance->registerAutoDescription<cluster::simple::Client>();
//    instance->registerAutoDescription<cluster::simple::Heartbeat>();
//    instance->registerAutoDescription<cluster::simple::Server>();
//    instance->registerAutoDescription<cluster::simple::View>();
//    instance->registerAutoDescription<cluster::mpi::MpiProvider>();
//    instance->registerAutoDescription<cluster::mpi::View>();
//    instance->registerAutoDescription<misc::ImageViewer>();
//    instance->registerAutoDescription<misc::LinesRenderer>();
//    instance->registerAutoDescription<misc::SiffCSplineFitter>();
//    instance->registerAutoDescription<misc::TestSpheresDataSource>();
//    instance->registerAutoDescription<moldyn::AddParticleColours>();
//    instance->registerAutoDescription<moldyn::ArrowRenderer>();
//    instance->registerAutoDescription<moldyn::DataGridder>();
//    instance->registerAutoDescription<moldyn::GrimRenderer>();
//    instance->registerDescription<LoaderADModuleAutoDescription<moldyn::IMDAtomDataSource> >();
//    instance->registerAutoDescription<moldyn::MipDepthSphereRenderer>();
//    instance->registerAutoDescription<moldyn::MMPGDDataSource>();
//    instance->registerAutoDescription<moldyn::MMPGDWriter>();
//    instance->registerAutoDescription<moldyn::MMPLDDataSource>();
//    instance->registerAutoDescription<moldyn::MMPLDWriter>();
//    instance->registerDescription<LoaderADModuleAutoDescription<moldyn::MMSPDDataSource> >();
//    instance->registerAutoDescription<moldyn::SIFFDataSource>();
//    instance->registerAutoDescription<moldyn::SimpleGeoSphereRenderer>();
//    instance->registerAutoDescription<moldyn::SimpleSphereRenderer>();
//    instance->registerAutoDescription<moldyn::NGSphereRenderer>();
//    instance->registerAutoDescription<moldyn::ClusteredSphereRenderer>();
//    instance->registerAutoDescription<moldyn::SphereOutlineRenderer>();
//    instance->registerAutoDescription<moldyn::OracleSphereRenderer>();
//    instance->registerAutoDescription<moldyn::VIMDataSource>();
//    instance->registerAutoDescription<moldyn::VisIttDataSource>();
//    instance->registerAutoDescription<moldyn::DirPartColModulate>();
//    instance->registerAutoDescription<moldyn::ParticleListFilter>();
//    instance->registerAutoDescription<moldyn::ParticleWorker>();
//    instance->registerAutoDescription<moldyn::DirPartFilter>();
//    //instance->registerAutoDescription<special::ColStereoDisplay>();
//    instance->registerAutoDescription<view::ClipPlane>();
//    instance->registerAutoDescription<view::LinearTransferFunction>();
//    instance->registerAutoDescription<view::TransferFunctionRenderer>();
//    instance->registerAutoDescription<view::MuxRenderer3D<2> >();
//    instance->registerAutoDescription<view::MuxRenderer3D<3> >();
//    instance->registerAutoDescription<view::MuxRenderer3D<4> >();
//    instance->registerAutoDescription<view::MuxRenderer3D<5> >();
//    instance->registerAutoDescription<view::MuxRenderer3D<10> >();
//    instance->registerAutoDescription<view::special::AnaglyphStereoView>();
//    instance->registerAutoDescription<view::special::ChronoGraph>();
//    instance->registerAutoDescription<view::special::DemoRenderer2D>();
//    instance->registerAutoDescription<view::special::QuadBufferStereoView>();
//    instance->registerAutoDescription<view::special::ScreenShooter>();
//    instance->registerAutoDescription<view::SwitchRenderer3D>();
//    instance->registerAutoDescription<view::TileView>();
//    instance->registerAutoDescription<view::View2D>();
//    instance->registerAutoDescription<view::View3D>();
//    instance->registerAutoDescription<view::ViewDirect3D>();
//    instance->registerAutoDescription<view::BlinnPhongRendererDeferred>();
//    instance->registerAutoDescription<view::SplitView>();
//    instance->registerAutoDescription<view::SharedCameraParameters>();
//    instance->registerAutoDescription<view::LinkedView3D>();
//    //instance->registerAutoDescription<vismol2::Mol20Renderer>();
//    //instance->registerAutoDescription<vismol2::Mol20DataSource>();
//    instance->registerAutoDescription<job::DataWriterJob>();
//    instance->registerAutoDescription<job::JobThread>();
//    instance->registerAutoDescription<moldyn::SIFFWriter>();
//    instance->registerAutoDescription<moldyn::VTFDataSource>();
//    instance->registerAutoDescription<moldyn::VTFResDataSource>();
//    instance->registerAutoDescription<moldyn::D3D11SimpleSphereRenderer>();
//    instance->registerAutoDescription<moldyn::AddClusterColours>();
//    instance->registerAutoDescription<moldyn::DynDensityGradientEstimator>();
//
//}
//
//
//
///*
// * ModuleDescriptionManager::ModuleDescriptionManager
// */
//ModuleDescriptionManager::ModuleDescriptionManager(void)
//        : ObjectDescriptionManager<ModuleDescription>() {
//    // intentionally empty
//}
//
//
///*
// * ModuleDescriptionManager::~ModuleDescriptionManager
// */
//ModuleDescriptionManager::~ModuleDescriptionManager(void) {
//    // intentionally empty
//}
//
//
///*
// * ModuleDescriptionManager::registerDescription
// */
//template<class Cp>
//void ModuleDescriptionManager::registerDescription(void) {
//    this->registerDescription(new Cp());
//}
//
//
///*
// * ModuleDescriptionManager::registerDescription
// */
//void ModuleDescriptionManager::registerDescription(
//        ModuleDescription* desc) {
//    ASSERT(desc != NULL);
//    // DO NOT test availability here! because we need all descriptions before
//    // OpenGL is started.
//    this->Register(desc);
//}
//
//
///*
// * ModuleDescriptionManager::registerAutoDescription
// */
//template<class Cp>
//void ModuleDescriptionManager::registerAutoDescription(void) {
//    this->registerDescription(new ModuleAutoDescription<Cp>());
//}
