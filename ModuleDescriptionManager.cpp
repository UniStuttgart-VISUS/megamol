/*
 * ModuleDescriptionManager.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ModuleDescriptionManager.h"
#include "LoaderADModuleAutoDescription.h"
#include "ModuleAutoDescription.h"
#include "ModuleDescription.h"
#include "vislib/assert.h"

#include "cluster/ClusterController.h"
#include "cluster/ClusterViewMaster.h"
#include "cluster/PowerwallView.h"
#include "cluster/mpi/MpiProvider.h"
#include "cluster/mpi/View.h"
#include "cluster/simple/Client.h"
#include "cluster/simple/Heartbeat.h"
#include "cluster/simple/Server.h"
#include "cluster/simple/View.h"
#include "misc/ImageViewer.h"
#include "misc/LinesRenderer.h"
#include "misc/SiffCSplineFitter.h"
#include "misc/TestSpheresDataSource.h"
#include "moldyn/AddParticleColours.h"
#include "moldyn/ArrowRenderer.h"
#include "moldyn/DataFileSequence.h"
#include "moldyn/DataGridder.h"
#include "moldyn/GrimRenderer.h"
#include "moldyn/IMDAtomDataSource.h"
#include "moldyn/MipDepthSphereRenderer.h"
#include "moldyn/MMPGDDataSource.h"
#include "moldyn/MMPGDWriter.h"
#include "moldyn/MMPLDDataSource.h"
#include "moldyn/MMPLDWriter.h"
#include "moldyn/MMSPDDataSource.h"
#include "moldyn/OracleSphereRenderer.h"
#include "moldyn/SIFFDataSource.h"
#include "moldyn/SimpleGeoSphereRenderer.h"
#include "moldyn/SimpleSphereRenderer.h"
#include "moldyn/NGSphereRenderer.h"
#include "moldyn/ClusteredSphereRenderer.h"
#include "moldyn/SphereOutlineRenderer.h"
#include "moldyn/VIMDataSource.h"
#include "moldyn/VisIttDataSource.h"
#include "moldyn/DirPartColModulate.h"
#include "moldyn/DirPartFilter.h"
#include "moldyn/ParticleListFilter.h"
#include "moldyn/ParticleWorker.h"
#include "moldyn/D3D11SimpleSphereRenderer.h"
//#include "special/ColStereoDisplay.h"
#include "view/ClipPlane.h"
#include "view/LinearTransferFunction.h"
#include "view/TransferFunctionRenderer.h"
#include "view/MuxRenderer3D.h"
#include "view/special/AnaglyphStereoView.h"
#include "view/special/ChronoGraph.h"
#include "view/special/DemoRenderer2D.h"
#include "view/special/QuadBufferStereoView.h"
#include "view/special/ScreenShooter.h"
#include "view/SwitchRenderer3D.h"
#include "view/TileView.h"
#include "view/View2D.h"
#include "view/View3D.h"
#include "view/ViewDirect3D.h"
#include "view/BlinnPhongRendererDeferred.h"
#include "view/SplitView.h"
#include "view/SharedCameraParameters.h"
#include "view/LinkedView3D.h"
#include "job/DataWriterJob.h"
#include "job/JobThread.h"
//#include "vismol2/Mol20DataSource.h"
//#include "vismol2/Mol20Renderer.h"
#include "BuckyBall.h"
#include "GridBalls.h"
#include "moldyn/DirPartVolume.h"
#include "misc/VolumeCache.h"
#include "RenderVolumeSlice.h"
#include "moldyn/DirectVolumeRenderer.h"
#include "moldyn/VolumeDataCall.h"
#include "moldyn/SIFFWriter.h"
#include "moldyn/VTFDataSource.h"
#include "moldyn/VTFResDataSource.h"
#include "moldyn/AddClusterColours.h"
#include "moldyn/DynDensityGradientEstimator.h"

using namespace megamol::core;


/*
 * ModuleDescriptionManager::inst
 */
vislib::SmartPtr<ModuleDescriptionManager> ModuleDescriptionManager::inst;


/*
 * ModuleDescriptionManager::Instance
 */
ModuleDescriptionManager * ModuleDescriptionManager::Instance() {
    if (inst.IsNull()) {
        inst = new ModuleDescriptionManager();
        registerObjects(inst.operator->());
    }
    return inst.operator->();
}


/*
 * ModuleDescriptionManager::ShutdownInstance
 */
void ModuleDescriptionManager::ShutdownInstance() {
    inst = NULL;
}


/*
 * ModuleDescriptionManager::registerObjects
 */
void ModuleDescriptionManager::registerObjects(ModuleDescriptionManager *instance) {

    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph module descriptions here
    //////////////////////////////////////////////////////////////////////

    instance->registerAutoDescription<cluster::ClusterController>();
    instance->registerAutoDescription<cluster::ClusterViewMaster>();
    instance->registerAutoDescription<cluster::PowerwallView>();
    instance->registerAutoDescription<cluster::simple::Client>();
    instance->registerAutoDescription<cluster::simple::Heartbeat>();
    instance->registerAutoDescription<cluster::simple::Server>();
    instance->registerAutoDescription<cluster::simple::View>();
    instance->registerAutoDescription<cluster::mpi::MpiProvider>();
    instance->registerAutoDescription<cluster::mpi::View>();
    instance->registerAutoDescription<misc::ImageViewer>();
    instance->registerAutoDescription<misc::LinesRenderer>();
    instance->registerAutoDescription<misc::SiffCSplineFitter>();
    instance->registerAutoDescription<misc::TestSpheresDataSource>();
    instance->registerAutoDescription<moldyn::AddParticleColours>();
    instance->registerAutoDescription<moldyn::ArrowRenderer>();
    instance->registerAutoDescription<moldyn::DataFileSequence>();
    instance->registerAutoDescription<moldyn::DataGridder>();
    instance->registerAutoDescription<moldyn::GrimRenderer>();
    instance->registerDescription<LoaderADModuleAutoDescription<moldyn::IMDAtomDataSource> >();
    instance->registerAutoDescription<moldyn::MipDepthSphereRenderer>();
    instance->registerAutoDescription<moldyn::MMPGDDataSource>();
    instance->registerAutoDescription<moldyn::MMPGDWriter>();
    instance->registerAutoDescription<moldyn::MMPLDDataSource>();
    instance->registerAutoDescription<moldyn::MMPLDWriter>();
    instance->registerDescription<LoaderADModuleAutoDescription<moldyn::MMSPDDataSource> >();
    instance->registerAutoDescription<moldyn::SIFFDataSource>();
    instance->registerAutoDescription<moldyn::SimpleGeoSphereRenderer>();
    instance->registerAutoDescription<moldyn::SimpleSphereRenderer>();
    instance->registerAutoDescription<moldyn::NGSphereRenderer>();
    instance->registerAutoDescription<moldyn::ClusteredSphereRenderer>();
    instance->registerAutoDescription<moldyn::SphereOutlineRenderer>();
    instance->registerAutoDescription<moldyn::OracleSphereRenderer>();
    instance->registerAutoDescription<moldyn::VIMDataSource>();
    instance->registerAutoDescription<moldyn::VisIttDataSource>();
    instance->registerAutoDescription<moldyn::DirPartColModulate>();
    instance->registerAutoDescription<moldyn::ParticleListFilter>();
    instance->registerAutoDescription<moldyn::ParticleWorker>();
    instance->registerAutoDescription<moldyn::DirPartFilter>();
    //instance->registerAutoDescription<special::ColStereoDisplay>();
    instance->registerAutoDescription<view::ClipPlane>();
    instance->registerAutoDescription<view::LinearTransferFunction>();
    instance->registerAutoDescription<view::TransferFunctionRenderer>();
    instance->registerAutoDescription<view::MuxRenderer3D<2> >();
    instance->registerAutoDescription<view::MuxRenderer3D<3> >();
    instance->registerAutoDescription<view::MuxRenderer3D<4> >();
    instance->registerAutoDescription<view::MuxRenderer3D<5> >();
    instance->registerAutoDescription<view::MuxRenderer3D<10> >();
    instance->registerAutoDescription<view::special::AnaglyphStereoView>();
    instance->registerAutoDescription<view::special::ChronoGraph>();
    instance->registerAutoDescription<view::special::DemoRenderer2D>();
    instance->registerAutoDescription<view::special::QuadBufferStereoView>();
    instance->registerAutoDescription<view::special::ScreenShooter>();
    instance->registerAutoDescription<view::SwitchRenderer3D>();
    instance->registerAutoDescription<view::TileView>();
    instance->registerAutoDescription<view::View2D>();
    instance->registerAutoDescription<view::View3D>();
    instance->registerAutoDescription<view::ViewDirect3D>();
    instance->registerAutoDescription<view::BlinnPhongRendererDeferred>();
    instance->registerAutoDescription<view::SplitView>();
    instance->registerAutoDescription<view::SharedCameraParameters>();
    instance->registerAutoDescription<view::LinkedView3D>();
    //instance->registerAutoDescription<vismol2::Mol20Renderer>();
    //instance->registerAutoDescription<vismol2::Mol20DataSource>();
    instance->registerAutoDescription<job::DataWriterJob>();
    instance->registerAutoDescription<job::JobThread>();
    instance->registerAutoDescription<BuckyBall>();
    instance->registerAutoDescription<GridBalls>();
    instance->registerAutoDescription<moldyn::DirPartVolume>();
    instance->registerAutoDescription<misc::VolumeCache>();
    instance->registerAutoDescription<RenderVolumeSlice>();
    instance->registerAutoDescription<moldyn::DirectVolumeRenderer>();
    instance->registerAutoDescription<moldyn::SIFFWriter>();
    instance->registerAutoDescription<moldyn::VTFDataSource>();
    instance->registerAutoDescription<moldyn::VTFResDataSource>();
    instance->registerAutoDescription<moldyn::D3D11SimpleSphereRenderer>();
    instance->registerAutoDescription<moldyn::AddClusterColours>();
    instance->registerAutoDescription<moldyn::DynDensityGradientEstimator>();

}



/*
 * ModuleDescriptionManager::ModuleDescriptionManager
 */
ModuleDescriptionManager::ModuleDescriptionManager(void)
        : ObjectDescriptionManager<ModuleDescription>() {
    // intentionally empty
}


/*
 * ModuleDescriptionManager::~ModuleDescriptionManager
 */
ModuleDescriptionManager::~ModuleDescriptionManager(void) {
    // intentionally empty
}


/*
 * ModuleDescriptionManager::registerDescription
 */
template<class Cp>
void ModuleDescriptionManager::registerDescription(void) {
    this->registerDescription(new Cp());
}


/*
 * ModuleDescriptionManager::registerDescription
 */
void ModuleDescriptionManager::registerDescription(
        ModuleDescription* desc) {
    ASSERT(desc != NULL);
    // DO NOT test availability here! because we need all descriptions before
    // OpenGL is started.
    this->Register(desc);
}


/*
 * ModuleDescriptionManager::registerAutoDescription
 */
template<class Cp>
void ModuleDescriptionManager::registerAutoDescription(void) {
    this->registerDescription(new ModuleAutoDescription<Cp>());
}
