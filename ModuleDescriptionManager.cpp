/*
 * ModuleDescriptionManager.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ModuleDescriptionManager.h"
#include "ModuleAutoDescription.h"
#include "ModuleDescription.h"
#include "vislib/assert.h"

#include "cluster/ClusterController.h"
#include "cluster/ClusterViewMaster.h"
#include "cluster/PowerwallView.h"
#include "DataFileSequencer.h"
#include "misc/BezierDataSource.h"
#include "misc/BezierMeshRenderer.h"
#include "misc/SiffCSplineFitter.h"
#include "moldyn/AddParticleColours.h"
#include "moldyn/DataGridder.h"
#include "moldyn/GrimRenderer.h"
#include "moldyn/IMDAtomDataSource.h"
#include "moldyn/MipDepthSphereRenderer.h"
#include "moldyn/MMPLDWriter.h"
#include "moldyn/OracleSphereRenderer.h"
#include "moldyn/SIFFDataSource.h"
#include "moldyn/SimpleSphereRenderer.h"
#include "moldyn/VIMDataSource.h"
//#include "special/AnaglyphStereoDisplay.h"
//#include "special/ColStereoDisplay.h"
#include "view/ClipPlane.h"
#include "view/LinearTransferFunction.h"
#include "view/special/ChronoGraph.h"
#include "view/special/DemoRenderer2D.h"
#include "view/special/QuadBufferStereoView.h"
#include "view/special/ScreenShooter.h"
#include "view/SwitchRenderer3D.h"
#include "view/TileView.h"
#include "view/View2D.h"
#include "view/View3D.h"
#include "job/DataWriterJob.h"
#include "job/JobThread.h"
//#include "vismol2/Mol20DataSource.h"
//#include "vismol2/Mol20Renderer.h"

using namespace megamol::core;


/*
 * ModuleDescriptionManager::Instance
 */
ModuleDescriptionManager *
ModuleDescriptionManager::Instance() {
    static ModuleDescriptionManager *instance = NULL;
    if (instance == NULL) {
        instance = new ModuleDescriptionManager();

        //////////////////////////////////////////////////////////////////////
        // Register all rendering graph module descriptions here
        //////////////////////////////////////////////////////////////////////

        instance->registerAutoDescription<cluster::ClusterController>();
        instance->registerAutoDescription<cluster::ClusterViewMaster>();
        instance->registerAutoDescription<cluster::PowerwallView>();
        instance->registerAutoDescription<DataFileSequencer>();
        instance->registerAutoDescription<misc::BezierDataSource>();
        instance->registerAutoDescription<misc::BezierMeshRenderer>();
        instance->registerAutoDescription<misc::SiffCSplineFitter>();
        instance->registerAutoDescription<moldyn::AddParticleColours>();
        instance->registerAutoDescription<moldyn::DataGridder>();
        instance->registerAutoDescription<moldyn::GrimRenderer>();
        instance->registerAutoDescription<moldyn::IMDAtomDataSource>();
        instance->registerAutoDescription<moldyn::MipDepthSphereRenderer>();
        instance->registerAutoDescription<moldyn::MMPLDWriter>();
        instance->registerAutoDescription<moldyn::SIFFDataSource>();
        instance->registerAutoDescription<moldyn::SimpleSphereRenderer>();
        instance->registerAutoDescription<moldyn::OracleSphereRenderer>();
        instance->registerAutoDescription<moldyn::VIMDataSource>();
        //instance->registerAutoDescription<special::AnaglyphStereoDisplay>();
        //instance->registerAutoDescription<special::ColStereoDisplay>();
        instance->registerAutoDescription<view::ClipPlane>();
        instance->registerAutoDescription<view::LinearTransferFunction>();
        instance->registerAutoDescription<view::special::ChronoGraph>();
        instance->registerAutoDescription<view::special::DemoRenderer2D>();
        instance->registerAutoDescription<view::special::QuadBufferStereoView>();
        instance->registerAutoDescription<view::special::ScreenShooter>();
        instance->registerAutoDescription<view::SwitchRenderer3D>();
        instance->registerAutoDescription<view::TileView>();
        instance->registerAutoDescription<view::View2D>();
        instance->registerAutoDescription<view::View3D>();
        //instance->registerAutoDescription<vismol2::Mol20Renderer>();
        //instance->registerAutoDescription<vismol2::Mol20DataSource>();
        instance->registerAutoDescription<job::DataWriterJob>();
        instance->registerAutoDescription<job::JobThread>();
    }
    return instance;
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
