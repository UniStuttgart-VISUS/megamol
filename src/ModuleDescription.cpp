/*
 * ModuleDescription.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/ModuleDescription.h"
#include "mmcore/CoreInstance.h"


using namespace megamol::core;


/*
 * ModuleDescription::ModuleDescription
 */
ModuleDescription::ModuleDescription(void) : ObjectDescription() {
}


/*
 * ModuleDescription::~ModuleDescription
 */
ModuleDescription::~ModuleDescription(void) {
}


/*
 * ModuleDescription::CreateModule
 */
Module *ModuleDescription::CreateModule(const vislib::StringA& name,
        class ::megamol::core::CoreInstance *instance) const {
    Module *m = this->createModuleImpl();
    if (m != NULL) {
        m->coreInst = instance;
        m->setModuleName(name);
    }
    return m;
}


/*
 * ModuleDescription::IsAvailable
 */
bool ModuleDescription::IsAvailable(void) const {
    return true;
}


/*
 * ModuleDescription::IsLoaderWithAutoDetection
 */
bool ModuleDescription::IsLoaderWithAutoDetection(void) const {
    return false;
}


/*
 * ModuleDescription::LoaderAutoDetectionFilenameExtensions
 */
const char *ModuleDescription::LoaderAutoDetectionFilenameExtensions(void) const {
    return NULL;
}


/*
 * ModuleDescription::LoaderAutoDetection
 */
float ModuleDescription::LoaderAutoDetection(const unsigned char* data, SIZE_T dataSize) const {
    return 0.0f;
}


/*
 * ModuleDescription::LoaderFilenameSlotName
 */
const char *ModuleDescription::LoaderFilenameSlotName(void) const {
    return NULL;
}


/*
 * ModuleDescription::LoaderFileTypeName
 */
const char *ModuleDescription::LoaderFileTypeName(void) const {
    return NULL;
}
