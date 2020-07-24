/*
 * GUI_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GUI_Service.hpp"

#include "vislib/sys/Log.h"


namespace megamol {
namespace frontend {


GUI_Service::~GUI_Service() {

}


bool GUI_Service::init(void* configPtr) {
    if (configPtr == nullptr) return false;

    return init(*static_cast<Config*>(configPtr));
}


bool GUI_Service::init(const Config& config) {
   
    return true;
}


void GUI_Service::close() {
 
}
	

void GUI_Service::updateProvidedResources() {
 
}


void GUI_Service::digestChangedRequestedResources() {
 
}


void GUI_Service::resetProvidedResources() {
    // nothing to do here
}


void GUI_Service::preGraphRender() {

}


void GUI_Service::postGraphRender() {

}


std::vector<ModuleResource>& GUI_Service::getProvidedResources() {
    // unused - returning empty list
	return m_providedResourceReferences;
}


const std::vector<std::string> GUI_Service::getRequestedResourceNames() const {
	return {
        {"MegaMolGraph"},
        {"KeyboardEvents"},
        {"MouseEvents"},
        {"IOpenGL_Context"}   
    };
}


void GUI_Service::setRequestedResources(std::vector<ModuleResource>& resources) {
    m_requestedResourceReferences = resources;
}


} // namespace frontend
} // namespace megamol
