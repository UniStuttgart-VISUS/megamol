/*
 * ClusterViewMaster.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterViewMaster.h"
#include "AbstractNamedObject.h"
#include "AbstractNamedObjectContainer.h"
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "ModuleNamespace.h"
#include "param/StringParam.h"
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/Log.h"
#include "vislib/StringTokeniser.h"

using namespace megamol::core;


/*
 * cluster::ClusterViewMaster::ClusterViewMaster
 */
cluster::ClusterViewMaster::ClusterViewMaster(void) : Module(),
        cluster::ClusterControllerClient(),
        viewNameSlot("viewname", "The name of the view to be used"),
        viewSlot("view", "The view to be used (this value is set automatically") {

    this->MakeSlotAvailable(&this->registerSlot);

    this->viewNameSlot << new param::StringParam("");
    this->viewNameSlot.SetUpdateCallback(&ClusterViewMaster::onViewNameChanged);
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->viewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    // TODO: this->viewSlot.SetVisibility(false);
    this->MakeSlotAvailable(&this->viewSlot);

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::~ClusterViewMaster
 */
cluster::ClusterViewMaster::~ClusterViewMaster(void) {
    this->Release();

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::create
 */
bool cluster::ClusterViewMaster::create(void) {

    // TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::release
 */
void cluster::ClusterViewMaster::release(void) {

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::onViewNameChanged
 */
bool cluster::ClusterViewMaster::onViewNameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    if (!this->viewSlot.ConnectCall(NULL)) { // disconnect old call
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Unable to disconnect call from slot \"%s\"\n",
            this->viewSlot.FullName().PeekBuffer());
    }

    CalleeSlot *viewModSlot = NULL;
    vislib::StringA viewName(this->viewNameSlot.Param<param::StringParam>()->Value());
    if (viewName.IsEmpty()) {
        // user just wanted to disconnect
        return true;
    }

    this->LockModuleGraph(false);
    AbstractNamedObject *ano = this->FindNamedObject(viewName);
    view::AbstractView *av = dynamic_cast<view::AbstractView*>(ano);
    if (av == NULL) {
        ModuleNamespace *mn = dynamic_cast<ModuleNamespace*>(ano);
        if (mn != NULL) {
            view::AbstractView *av2;
            AbstractNamedObjectContainer::ChildList::Iterator ci = mn->GetChildIterator();
            while (ci.HasNext()) {
                ano = ci.Next();
                av2 = dynamic_cast<view::AbstractView*>(ano);
                if (av2 != NULL) {
                    if (av != NULL) {
                        av = NULL;
                        break; // too many views
                    } else {
                        av = av2; // if only one view present in children, use it
                    }
                }
            }
        }
    }
    if (av != NULL) {
        viewModSlot = dynamic_cast<CalleeSlot*>(av->FindSlot("render"));
    }
    this->UnlockModuleGraph();

    if (viewModSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "View \"%s\" not found\n",
            viewName.PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    CallDescription *cd = CallDescriptionManager::Instance()
        ->Find(view::CallRenderView::ClassName());
    if (cd == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find description for call \"%s\"\n",
            view::CallRenderView::ClassName());
        return true; // this is just for diryt flag reset
    }

    Call *c = cd->CreateCall();
    if (c == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot create call \"%s\"\n",
            view::CallRenderView::ClassName());
        return true; // this is just for diryt flag reset
    }

    if (!viewModSlot->ConnectCall(c)) {
        delete c;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot connect call \"%s\" to inbound-slot \"%s\"\n",
            view::CallRenderView::ClassName(),
            viewModSlot->FullName().PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    if (!this->viewSlot.ConnectCall(c)) {
        delete c;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot connect call \"%s\" to outbound-slot \"%s\"\n",
            view::CallRenderView::ClassName(),
            this->viewSlot.FullName().PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    // TODO: Implement

    return true;
}
