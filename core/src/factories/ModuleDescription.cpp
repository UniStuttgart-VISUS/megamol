/*
 * ModuleDescription.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/Module.h"


using namespace megamol::core;


/*
 * factories::ModuleDescription::ModuleDescription
 */
factories::ModuleDescription::ModuleDescription(void) : ObjectDescription() {
}


/*
 * factories::ModuleDescription::~ModuleDescription
 */
factories::ModuleDescription::~ModuleDescription(void) {
}


/*
 * factories::ModuleDescription::CreateModule
 */
Module::ptr_type factories::ModuleDescription::CreateModule(const vislib::StringA& name) const {
    Module::ptr_type m = this->createModuleImpl();
    if (m) {
        m->fixParentBackreferences();
        m->setModuleName(name);
    }
    return m;
}


/*
 * factories::ModuleDescription::IsAvailable
 */
bool factories::ModuleDescription::IsAvailable(void) const {
    return true;
}


/*
 * factories::ModuleDescription::IsLoaderWithAutoDetection
 */
bool factories::ModuleDescription::IsLoaderWithAutoDetection(void) const {
    return false;
}


/*
 * factories::ModuleDescription::LoaderAutoDetectionFilenameExtensions
 */
const char *factories::ModuleDescription::LoaderAutoDetectionFilenameExtensions(void) const {
    return nullptr;
}


/*
 * factories::ModuleDescription::LoaderAutoDetection
 */
float factories::ModuleDescription::LoaderAutoDetection(const unsigned char* data, SIZE_T dataSize) const {
    return 0.0f;
}


/*
 * factories::ModuleDescription::LoaderFilenameSlotName
 */
const char *factories::ModuleDescription::LoaderFilenameSlotName(void) const {
    return nullptr;
}


/*
 * factories::ModuleDescription::LoaderFileTypeName
 */
const char *factories::ModuleDescription::LoaderFileTypeName(void) const {
    return nullptr;
}
