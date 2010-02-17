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

#include "misc/BezierDataSource.h"
#include "misc/BezierMeshRenderer.h"
#include "misc/DemoRenderer2D.h"
#include "moldyn/AddParticleColours.h"
#include "moldyn/DataGridder.h"
#include "moldyn/GrimRenderer.h"
#include "moldyn/IMDAtomDataSource.h"
#include "moldyn/MipDepthSphereRenderer.h"
#include "moldyn/OracleSphereRenderer.h"
#include "moldyn/SIFFDataSource.h"
#include "moldyn/SimpleSphereRenderer.h"
#include "moldyn/VIMDataSource.h"
//#include "special/AnaglyphStereoDisplay.h"
//#include "special/ClusterController.h"
//#include "special/ClusterDisplay.h"
//#include "special/ColStereoDisplay.h"
//#include "special/RenderMaster.h"
//#include "special/TimeView.h"
//#include "special/TitleSceneView.h"
//#include "special/ScreenShooter.h"
//#include "special/VisLogoRenderer.h"
//#include "volume/VolumeRenderer.h"
//#include "volume/VolumeData.h"
//#include "volume/Scatterplot2D.h"
//#include "volume/ContinuousParallelCoordinates.h"
//#include "volume/View2D.h"
#include "view/ClipPlane.h"
#include "view/LinearTransferFunction.h"
#include "view/OverrideView.h"
#include "view/SwitchRenderer3D.h"
#include "view/View2D.h"
#include "view/View3D.h"
#include "job/JobThread.h"
//#include "misc/TriSoupRenderer.h"
//#include "vismol2/Mol20DataSource.h"
//#include "vismol2/Mol20Renderer.h"

//#include "featuretracking/DataSource.h"
//#include "featuretracking/Renderer1.h"
//#include "featuretracking/Renderer2.h"
//#include "featuretracking/LinkingBrushing.h"
//#include "featuretracking/Extractor.h"

//#include "protein/ProteinRendererCartoon.h"
//#include "protein/ProteinRenderer.h"
//#include "protein/ProteinData.h"
//#include "protein/NetCDFData.h"
//#include "protein/RMSRenderer.h"

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

        instance->registerAutoDescription<misc::BezierDataSource>();
        instance->registerAutoDescription<misc::BezierMeshRenderer>();
        instance->registerAutoDescription<misc::DemoRenderer2D>();
        instance->registerAutoDescription<moldyn::AddParticleColours>();
        instance->registerAutoDescription<moldyn::DataGridder>();
        instance->registerAutoDescription<moldyn::GrimRenderer>();
        instance->registerAutoDescription<moldyn::IMDAtomDataSource>();
        instance->registerAutoDescription<moldyn::MipDepthSphereRenderer>();
        instance->registerAutoDescription<moldyn::SIFFDataSource>();
        instance->registerAutoDescription<moldyn::SimpleSphereRenderer>();
        instance->registerAutoDescription<moldyn::OracleSphereRenderer>();
        instance->registerAutoDescription<moldyn::VIMDataSource>();
        //instance->registerAutoDescription<special::AnaglyphStereoDisplay>();
        //instance->registerAutoDescription<special::ClusterController>();
        //instance->registerAutoDescription<special::ClusterDisplay>();
        //instance->registerAutoDescription<special::ColStereoDisplay>();
        //instance->registerAutoDescription<special::RenderMaster>();
        //instance->registerAutoDescription<special::ScreenShooter>();
        //instance->registerAutoDescription<special::TimeView>();
        //instance->registerAutoDescription<special::TitleSceneView>();
        instance->registerAutoDescription<view::ClipPlane>();
        instance->registerAutoDescription<view::LinearTransferFunction>();
        instance->registerAutoDescription<view::OverrideView>();
        instance->registerAutoDescription<view::SwitchRenderer3D>();
        instance->registerAutoDescription<view::View2D>();
        instance->registerAutoDescription<view::View3D>();

        instance->registerAutoDescription<job::JobThread>();
        //instance->registerAutoDescription<special::VisLogoRenderer>();
//        instance->registerAutoDescription<volume::VolumeRenderer>();
//        instance->registerAutoDescription<volume::VolumeData>();
//        instance->registerAutoDescription<volume::Scatterplot2D>();
//        instance->registerAutoDescription<volume::View2D>();
        //instance->registerAutoDescription<misc::TriSoupRenderer>();
//        instance->registerAutoDescription<featuretracking::LinkingBrushing>();
//        instance->registerAutoDescription<featuretracking::DataSource>();
//        instance->registerAutoDescription<featuretracking::Renderer1>();
//        instance->registerAutoDescription<featuretracking::Renderer2>();
//		instance->registerAutoDescription<featuretracking::Extractor>();
//        instance->registerAutoDescription<volume::ContinuousParallelCoordinates>();
        //instance->registerAutoDescription<vismol2::Mol20Renderer>();
        //instance->registerAutoDescription<vismol2::Mol20DataSource>();

        //instance->registerAutoDescription<protein::ProteinRendererCartoon>();
        //instance->registerAutoDescription<protein::ProteinRenderer>();
        //instance->registerAutoDescription<protein::ProteinData>();
//        instance->registerAutoDescription<protein::RMSRenderer>();
//
//#if (defined(WITH_NETCDF) && (WITH_NETCDF))
//		instance->registerAutoDescription<protein::NetCDFData>();
//#endif /* (defined(WITH_NETCDF) && (WITH_NETCDF)) */

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
