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

#include "misc/BezierDataCall.h"
#include "moldyn/MultiParticleDataCall.h"
#include "moldyn/ParticleGridDataCall.h"
#include "cluster/CallRegisterAtController.h"
#include "view/CallClipPlane.h"
#include "view/CallGetTransferFunction.h"
#include "view/CallRender2D.h"
#include "view/CallRender3D.h"
#include "view/CallRenderView.h"
//#include "volume/CallVolumeData.h"
//
//#include "featuretracking/CallGetData.h"
//#include "featuretracking/CallLinkingBrushing.h"

//#include "protein/CallProteinData.h"
//#include "protein/CallFrame.h"
//#include "vismol2/Mol20DataCall.h"

using namespace megamol::core;


/*
 * CallDescriptionManager::Instance
 */
CallDescriptionManager *
CallDescriptionManager::Instance() {
    static CallDescriptionManager *instance = NULL;
    if (instance == NULL) {
        instance = new CallDescriptionManager();

        //////////////////////////////////////////////////////////////////////
        // Register all rendering graph call descriptions here
        //////////////////////////////////////////////////////////////////////
        instance->registerAutoDescription<cluster::CallRegisterAtController>();
        instance->registerAutoDescription<misc::BezierDataCall>();
        instance->registerAutoDescription<moldyn::MultiParticleDataCall>();
        instance->registerAutoDescription<moldyn::ParticleGridDataCall>();
        instance->registerAutoDescription<view::CallClipPlane>();
        instance->registerAutoDescription<view::CallGetTransferFunction>();
        instance->registerAutoDescription<view::CallRender2D>();
        instance->registerAutoDescription<view::CallRender3D>();
        instance->registerAutoDescription<view::CallRenderView>();
//        instance->registerAutoDescription<vismol2::Mol20DataCall>();

//		instance->registerAutoDescription<volume::CallVolumeData>();

//		instance->registerAutoDescription<featuretracking::CallGetData>();
//		instance->registerAutoDescription<featuretracking::CallLinkingBrushing>();

        //instance->registerAutoDescription<protein::CallProteinData>();
        //instance->registerAutoDescription<protein::CallFrame>();
    }
    return instance;
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
