/*
 * CallDescriptionManager.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallDescriptionManager.h"
#include "CallAutoDescription.h"
#include "CallDescription.h"
#include "vislib/assert.h"

#include "misc/LinesDataCall.h"
#include "DataWriterCtrlCall.h"
#include "moldyn/DirectionalParticleDataCall.h"
#include "moldyn/MultiParticleDataCall.h"
#include "moldyn/ParticleGridDataCall.h"
#include "cluster/CallRegisterAtController.h"
#include "cluster/simple/ClientViewRegistration.h"
#include "view/CallClipPlane.h"
#include "view/CallGetTransferFunction.h"
#include "view/CallRender2D.h"
#include "view/CallRender3D.h"
#include "view/CallRenderDeferred3D.h"
#include "view/CallRenderView.h"
#include "view/CallTimeControl.h"
#include "view/CallUpdateDirect3D.h"
#include "CallVolumeData.h"
#include "moldyn/VolumeDataCall.h"
#include "misc/BezierCurvesListDataCall.h"
#include "misc/VolumetricDataCall.h"

using namespace megamol::core;


vislib::SmartPtr<CallDescriptionManager> CallDescriptionManager::inst;


/*
 * CallDescriptionManager::Instance
 */
CallDescriptionManager * CallDescriptionManager::Instance() {
    if (inst.IsNull()) {
        inst = new CallDescriptionManager();
        registerObjects(inst.operator->());
    }
    return inst.operator->();
}


void CallDescriptionManager::ShutdownInstance() {
    inst = NULL;
}


void CallDescriptionManager::registerObjects(CallDescriptionManager *instance) {
    //static CallDescriptionManager *instance = NULL;
    //if (instance == NULL) {
        //instance = new CallDescriptionManager();

        //////////////////////////////////////////////////////////////////////
        // Register all rendering graph call descriptions here
        //////////////////////////////////////////////////////////////////////
        instance->registerAutoDescription<cluster::CallRegisterAtController>();
        instance->registerAutoDescription<cluster::simple::ClientViewRegistration>();
        instance->registerAutoDescription<misc::LinesDataCall>();
        instance->registerAutoDescription<moldyn::DirectionalParticleDataCall>();
        instance->registerAutoDescription<moldyn::MultiParticleDataCall>();
        instance->registerAutoDescription<moldyn::ParticleGridDataCall>();
        instance->registerAutoDescription<view::CallClipPlane>();
        instance->registerAutoDescription<view::CallGetTransferFunction>();
        instance->registerAutoDescription<view::CallRender2D>();
        instance->registerAutoDescription<view::CallRender3D>();
        instance->registerAutoDescription<view::CallRenderDeferred3D>();
        instance->registerAutoDescription<view::CallRenderView>();
        instance->registerAutoDescription<view::CallTimeControl>();
        instance->registerAutoDescription<DataWriterCtrlCall>();
        instance->registerAutoDescription<CallVolumeData>();
        instance->registerAutoDescription<moldyn::VolumeDataCall>();
        instance->registerAutoDescription<misc::BezierCurvesListDataCall>();
        instance->registerAutoDescription<misc::VolumetricDataCall>();
        instance->registerAutoDescription<view::CallUpdateDirect3D>();
    //}
    //return instance;
}


/*
 * CallDescriptionManager::AssignmentCrowbar
 */
void CallDescriptionManager::AssignmentCrowbar(Call *tar, Call *src) {
    DescriptionIterator iter = this->GetIterator();
    while (iter.HasNext()) {
        const CallDescription *desc = iter.Next();
        if (desc->IsDescribing(tar)) {
            if (desc->IsDescribing(src)) {
                desc->AssignmentCrowbar(tar, src);
            } else {
                // TODO: ARGLHARGLGARGLGARG
            }
        }
    }
}


/*
 * CallDescriptionManager::CallDescriptionManager
 */
CallDescriptionManager::CallDescriptionManager(void)
        : ObjectDescriptionManager<CallDescription>() {
    // intentionally empty
}


/*
 * CallDescriptionManager::~CallDescriptionManager
 */
CallDescriptionManager::~CallDescriptionManager(void) {
    // intentionally empty
}


/*
 * view::CallDescriptionManager::registerDescription
 */
template<class Cp>
void CallDescriptionManager::registerDescription(void) {
    this->registerDescription(new Cp());
}


/*
 * CallDescriptionManager::registerDescription
 */
void CallDescriptionManager::registerDescription(
        CallDescription* desc) {
    ASSERT(desc != NULL);
    // DO NOT test availability here! because we need all descriptions before
    // OpenGL is started.
    this->Register(desc);
}


/*
 * CallDescriptionManager::registerAutoDescription
 */
template<class Cp>
void CallDescriptionManager::registerAutoDescription(void) {
    this->registerDescription(new CallAutoDescription<Cp>());
}
