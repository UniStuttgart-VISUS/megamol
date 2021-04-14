/*
 * Template_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// search/replace Template_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "Template_Service.hpp"


// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "Template_Service: ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}


namespace megamol {
namespace frontend {

Template_Service::Template_Service() {
    // init members to default states
}

Template_Service::~Template_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does 
}

bool Template_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool Template_Service::init(const Config& config) {
    // initialize your service and its provided resources using config parameters
    // for now, you dont need to worry about your service beeing initialized or closed multiple times
    // init() and close() only get called once in the lifetime of each service object
    // but maybe more instances of your service will get created? this may be relevant for central resources you manage (like libraries, network connections).

    if (init_failed) {
        log_error("failed initialization because");
        return false;
    }

    log("initialized successfully");
    return true;
}

void Template_Service::close() {
    // close libraries or APIs you manage
    // wrap up resources your service provides, but don not depend on outside resources to be available here
    // after this, at some point only the destructor of your service gets called
}

std::vector<FrontendResource>& Template_Service::getProvidedResources() {
    this->m_providedResource1 = MyProvidedResource_1{...};
    this->m_providedResource2 = MyProvidedResource_2{...};
    this->m_providedResource3 = MyProvidedResource_3{...};

    this->m_providedResourceReferences =
    { // construct std::vector
        {"MyProvidedResource_1", m_providedResource1}, // constructor FrontendResource{"MyProvidedResource_1", m_providedResource1}
        {"MyProvidedResource_2", m_providedResource2 /*reference to resource gets passed around*/ },
        {"MyProvidedResource_3" /*resources are identified using unique names in the system*/ , m_providedResource3}
    };

    return m_providedResourceReferences;
}

const std::vector<std::string> Template_Service::getRequestedResourceNames() const {
    // since this function should not change the state of the service
    // you should assign your requested resource names in init()
    this->m_requestedResourcesNames =
    {
        "ExternalResource_1",
        "ExternalResource_2"
    };

    return m_requestedResourcesNames;

    // alternative
    return 
    {
        "ExternalResource_1",
        "ExternalResource_2"
    };
}

void Template_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    // maybe we want to keep the list of requested resources
    this->m_requestedResourceReferences = resources;

    // prepare usage of requested resources
    this->m_externalResource_1_ptr = &resources[0].getResource<ExternalResource_1>(); // resources are in requested order and not null
    this->m_externalResource_2_ptr = &resources[1].getResource<namspace::to::resource::ExternalResource_2>(); // ptr will be not null or program terminates by design
}
    
void Template_Service::updateProvidedResources() {
    // update resources we provide to others with new available data

    this->m_providedResource1.update();
    this->m_providedResource2 = MyProvidedResource_2{new_data()};

    // deleting resources others may be using is not good
    // you need to guarantee that your resource objects are alive and usable until your close() gets called
    delete this->m_providedResource3; // DONT DO THIS
}

void Template_Service::digestChangedRequestedResources() {
    digest_changes(*this->m_externalResource_1_ptr); // not that the pointer should never become invalid by design
    digest_changes(*this->m_externalResource_2_ptr); // not that the pointer should never become invalid by design

    // FrontendResource::getResource<>() returns CONST references. if you know what you are doing you may modify resources that are not yours.
    modify_resource(const_cast<ExternalResource_1&>( resources[0].getResource<ExternalResource_1>() ));

    if (need_to_shutdown)
        this->setShutdown();
}

void Template_Service::resetProvidedResources() {
    // this gets called at the end of the main loop iteration
    // since the current resources state should have been handled in this frame already 
    // you may clean up resources whose state is not needed for the next iteration
    // e.g. m_keyboardEvents.clear();
    // network_traffic_buffer.reset_to_empty();
}

void Template_Service::preGraphRender() {
    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void Template_Service::postGraphRender() {
    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}


} // namespace frontend
} // namespace megamol
