/*
 * AbstractAssemblyInstance.cpp
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/AbstractAssemblyInstance.h"
#include "vislib/UTF8Encoder.h"


using namespace megamol::core::factories;


/*
 * AbstractAssemblyInstance::GetCallDescriptionManager
 */
const CallDescriptionManager&
AbstractAssemblyInstance::GetCallDescriptionManager(void) const {
    return this->call_descriptions;
}


/*
 * AbstractAssemblyInstance::GetModuleDescriptionManager
 */
const ModuleDescriptionManager&
AbstractAssemblyInstance::GetModuleDescriptionManager(void) const {
    return this->module_descriptions;
}

/*
 * AbstractAssemblyInstance::GetAssemblyFileName
 */
void AbstractAssemblyInstance::GetAssemblyFileName(std::string& out_filename) const {
    out_filename.clear();
    if (filename == nullptr) return;
    vislib::StringA s;
    vislib::UTF8Encoder::Decode(s, filename);
    out_filename = s;
}

/*
 * AbstractAssemblyInstance::GetAssemblyFileName
 */
void AbstractAssemblyInstance::GetAssemblyFileName(std::wstring& out_filename) const {
    out_filename.clear();
    if (filename == nullptr) return;
    vislib::StringW s;
    vislib::UTF8Encoder::Decode(s, filename);
    out_filename = s;
}

/*
 * AbstractAssemblyInstance::SetAssemblyFileName
 */
void AbstractAssemblyInstance::SetAssemblyFileName(const std::string& filename) {
    vislib::StringA u;
    vislib::UTF8Encoder::Encode(u, filename.c_str());
    if (this->filename != nullptr) delete[] this->filename;
    size_t l = u.Length();
    this->filename = new char[l + 1];
    ::memcpy(this->filename, u.PeekBuffer(), l);
    this->filename[l] = 0;
}

/*
 * AbstractAssemblyInstance::SetAssemblyFileName
 */
void AbstractAssemblyInstance::SetAssemblyFileName(const std::wstring& filename) {
    vislib::StringA u;
    vislib::UTF8Encoder::Encode(u, filename.c_str());
    if (this->filename != nullptr) delete[] this->filename;
    size_t l = u.Length();
    this->filename = new char[l + 1];
    ::memcpy(this->filename, u.PeekBuffer(), l);
    this->filename[l] = 0;
}

/*
 * AbstractAssemblyInstance::AbstractAssemblyInstance
 */
AbstractAssemblyInstance::AbstractAssemblyInstance(void) 
        : call_descriptions(), module_descriptions(), filename(nullptr) {
    // intentionally empty
}


/*
 * AbstractAssemblyInstance::~AbstractAssemblyInstance
 */
AbstractAssemblyInstance::~AbstractAssemblyInstance(void) {
    if (filename != nullptr) {
        delete[] filename;
        filename = nullptr;
    }
}
