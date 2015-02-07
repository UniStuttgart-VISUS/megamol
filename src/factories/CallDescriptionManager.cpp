/*
 * CallDescriptionManager.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/CallDescriptionManager.h"


/*
 * megamol::core::factories::CallDescriptionManager::CallDescriptionManager
 */
megamol::core::factories::CallDescriptionManager::CallDescriptionManager()
        : ObjectDescriptionManager<megamol::core::factories::CallDescription>() {
    // intentionally empty
}


/*
 * megamol::core::factories::CallDescriptionManager::~CallDescriptionManager
 */
megamol::core::factories::CallDescriptionManager::~CallDescriptionManager() {
    // intentionally empty
}


/*
 * megamol::core::factories::CallDescriptionManager::AssignmentCrowbar
 */
bool megamol::core::factories::CallDescriptionManager::AssignmentCrowbar(
        megamol::core::Call *tar, megamol::core::Call *src) const {
    for (auto desc : *this) {
        if (desc->IsDescribing(tar)) {
            if (desc->IsDescribing(src)) {
                desc->AssignmentCrowbar(tar, src);
                return true;
            } else {
                // TODO: ARGLHARGLGARGLGARG
            }
        }
    }
    return false;
}


//#include "mmcore/factories/CallAutoDescription.h"
//#include "mmcore/factories/CallDescription.h"
//#include "vislib/assert.h"
//
//#include "mmcore/misc/LinesDataCall.h"
//#include "mmcore/DataWriterCtrlCall.h"
//#include "mmcore/moldyn/DirectionalParticleDataCall.h"
//#include "mmcore/moldyn/MultiParticleDataCall.h"
//#include "mmcore/moldyn/ParticleGridDataCall.h"
//#include "mmcore/cluster/CallRegisterAtController.h"
//#include "mmcore/cluster/simple/ClientViewRegistration.h"
//#include "mmcore/view/CallClipPlane.h"
//#include "mmcore/view/CallGetTransferFunction.h"
//#include "mmcore/view/CallRender2D.h"
//#include "mmcore/view/CallRender3D.h"
//#include "mmcore/view/CallRenderDeferred3D.h"
//#include "mmcore/view/CallRenderView.h"
//#include "mmcore/view/CallTimeControl.h"
//#include "mmcore/view/CallUpdateDirect3D.h"
//#include "mmcore/view/CallCamParams.h"
//#include "mmcore/view/CallCamParamSync.h"
//#include "mmcore/CallVolumeData.h"
//#include "mmcore/moldyn/VolumeDataCall.h"
//#include "mmcore/misc/BezierCurvesListDataCall.h"
//#include "mmcore/misc/VolumetricDataCall.h"
//#include "mmcore/misc/QRCodeDataCall.h"
//#include "mmcore/misc/CalloutImageCall.h"
//#include "mmcore/view/Call6DofInteraction.h"
//#include "mmcore/cluster/mpi/MpiCall.h"
//
//using namespace megamol::core;
//
//
//vislib::SmartPtr<CallDescriptionManager> CallDescriptionManager::inst;
//
//
///*
// * CallDescriptionManager::Instance
// */
//CallDescriptionManager * CallDescriptionManager::Instance() {
//    if (inst.IsNull()) {
//        inst = new CallDescriptionManager();
//        registerObjects(inst.operator->());
//    }
//    return inst.operator->();
//}
//
//
//void CallDescriptionManager::ShutdownInstance() {
//    inst = NULL;
//}
//
//
//void CallDescriptionManager::registerObjects(CallDescriptionManager *instance) {
//    //static CallDescriptionManager *instance = NULL;
//    //if (instance == NULL) {
//        //instance = new CallDescriptionManager();
//
//        //////////////////////////////////////////////////////////////////////
//        // Register all rendering graph call descriptions here
//        //////////////////////////////////////////////////////////////////////
//        instance->registerAutoDescription<cluster::CallRegisterAtController>();
//        instance->registerAutoDescription<cluster::simple::ClientViewRegistration>();
//        instance->registerAutoDescription<misc::LinesDataCall>();
//        instance->registerAutoDescription<moldyn::DirectionalParticleDataCall>();
//        instance->registerAutoDescription<moldyn::MultiParticleDataCall>();
//        instance->registerAutoDescription<moldyn::ParticleGridDataCall>();
//        instance->registerAutoDescription<view::CallClipPlane>();
//        instance->registerAutoDescription<view::CallGetTransferFunction>();
//        instance->registerAutoDescription<view::CallRender2D>();
//        instance->registerAutoDescription<view::CallRender3D>();
//        instance->registerAutoDescription<view::CallRenderDeferred3D>();
//        instance->registerAutoDescription<view::CallRenderView>();
//        instance->registerAutoDescription<view::CallTimeControl>();
//        instance->registerAutoDescription<DataWriterCtrlCall>();
//        instance->registerAutoDescription<CallVolumeData>();
//        instance->registerAutoDescription<moldyn::VolumeDataCall>();
//        instance->registerAutoDescription<misc::BezierCurvesListDataCall>();
//        instance->registerAutoDescription<misc::VolumetricDataCall>();
//        instance->registerAutoDescription<view::CallUpdateDirect3D>();
//        instance->registerAutoDescription<view::CallCamParams>();
//        instance->registerAutoDescription<view::CallCamParamSync>();
//        instance->registerAutoDescription<misc::QRCodeDataCall>();
//        instance->registerAutoDescription<misc::CalloutImageCall>();
//        instance->registerAutoDescription<view::Call6DofInteraction>();
//        instance->registerAutoDescription<cluster::mpi::MpiCall>();
//    //}
//    //return instance;
//}
//
//
//
//
///*
// * CallDescriptionManager::CallDescriptionManager
// */
//CallDescriptionManager::CallDescriptionManager(void)
//        : ObjectDescriptionManager<CallDescription>() {
//    // intentionally empty
//}
//
//
///*
// * CallDescriptionManager::~CallDescriptionManager
// */
//CallDescriptionManager::~CallDescriptionManager(void) {
//    // intentionally empty
//}
//
//
///*
// * view::CallDescriptionManager::registerDescription
// */
//template<class Cp>
//void CallDescriptionManager::registerDescription(void) {
//    this->registerDescription(new Cp());
//}
//
//
///*
// * CallDescriptionManager::registerDescription
// */
//void CallDescriptionManager::registerDescription(
//        CallDescription* desc) {
//    ASSERT(desc != NULL);
//    // DO NOT test availability here! because we need all descriptions before
//    // OpenGL is started.
//    this->Register(desc);
//}
//
//
///*
// * CallDescriptionManager::registerAutoDescription
// */
//template<class Cp>
//void CallDescriptionManager::registerAutoDescription(void) {
//    this->registerDescription(new CallAutoDescription<Cp>());
//}
