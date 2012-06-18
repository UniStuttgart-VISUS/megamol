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
#include "cluster/simple/Client.h"
#include "cluster/simple/Heartbeat.h"
#include "cluster/simple/Server.h"
#include "cluster/simple/View.h"
#include "DataFileSequencer.h"
#include "misc/BezierControlLines.h"
#include "misc/BezierDataSource.h"
#include "misc/BezierMeshRenderer.h"
#include "misc/BezierRaycastRenderer.h"
#include "misc/ExtBezierDataSource.h"
#include "misc/ExtBezierMeshRenderer.h"
#include "misc/ExtBezierRaycastRenderer.h"
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
#include "moldyn/SimpleSphereRenderer.h"
#include "moldyn/SphereOutlineRenderer.h"
#include "moldyn/VIMDataSource.h"
#include "moldyn/VisIttDataSource.h"
#include "moldyn/DirPartColModulate.h"
#include "moldyn/DirPartFilter.h"
//#include "special/ColStereoDisplay.h"
#include "view/ClipPlane.h"
#include "view/LinearTransferFunction.h"
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
#include "view/BlinnPhongRendererDeferred.h"
#include "job/DataWriterJob.h"
#include "job/JobThread.h"
//#include "vismol2/Mol20DataSource.h"
//#include "vismol2/Mol20Renderer.h"
#include "BuckyBall.h"
#include "GridBalls.h"
#include "moldyn/DirPartVolume.h"
#include "misc/VolumeCache.h"
#include "RenderVolumeSlice.h"

using namespace megamol::core;


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


void ModuleDescriptionManager::ShutdownInstance() {
    inst = NULL;
}


void ModuleDescriptionManager::registerObjects(ModuleDescriptionManager *instance) {
    //static ModuleDescriptionManager *instance = NULL;
    //if (instance == NULL) {
        //instance = new ModuleDescriptionManager();

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
        instance->registerAutoDescription<DataFileSequencer>();
        instance->registerAutoDescription<misc::BezierControlLines>();
        instance->registerAutoDescription<misc::BezierDataSource>();
        instance->registerAutoDescription<misc::BezierMeshRenderer>();
        instance->registerAutoDescription<misc::BezierRaycastRenderer>();
        instance->registerAutoDescription<misc::ExtBezierDataSource>();
        instance->registerAutoDescription<misc::ExtBezierMeshRenderer>();
        instance->registerAutoDescription<misc::ExtBezierRaycastRenderer>();
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
        instance->registerAutoDescription<moldyn::SimpleSphereRenderer>();
        instance->registerAutoDescription<moldyn::SphereOutlineRenderer>();
        instance->registerAutoDescription<moldyn::OracleSphereRenderer>();
        instance->registerAutoDescription<moldyn::VIMDataSource>();
        instance->registerAutoDescription<moldyn::VisIttDataSource>();
        instance->registerAutoDescription<moldyn::DirPartColModulate>();
        instance->registerAutoDescription<moldyn::DirPartFilter>();
        //instance->registerAutoDescription<special::ColStereoDisplay>();
        instance->registerAutoDescription<view::ClipPlane>();
        instance->registerAutoDescription<view::LinearTransferFunction>();
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
        instance->registerAutoDescription<view::BlinnPhongRendererDeferred>();
        //instance->registerAutoDescription<vismol2::Mol20Renderer>();
        //instance->registerAutoDescription<vismol2::Mol20DataSource>();
        instance->registerAutoDescription<job::DataWriterJob>();
        instance->registerAutoDescription<job::JobThread>();
        instance->registerAutoDescription<BuckyBall>();
        instance->registerAutoDescription<GridBalls>();
        instance->registerAutoDescription<moldyn::DirPartVolume>();
        instance->registerAutoDescription<misc::VolumeCache>();
        instance->registerAutoDescription<RenderVolumeSlice>();
    //}
    //return instance;
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
