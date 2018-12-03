/*
* CallADIOSData.cpp
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallADIOSData.h"
#include "vislib/sys/Log.h"
#include <algorithm>

namespace megamol {
namespace adios {
    
CallADIOSData::CallADIOSData() : dataptr(nullptr)
    , dataHash(0)
    , inqVars()
    , availableVars()
    , frameIDtoLoad(0)
    , frameCount(0) { 
    // empty
}

CallADIOSData::~CallADIOSData(void) {}

void CallADIOSData::setTime(float time) { this->time = time; }

float CallADIOSData::getTime() { return this->time; }

void CallADIOSData::inquire(std::string varname) {
    if (!this->availableVars.empty()) {
        if (std::find(this->availableVars.begin(), this->availableVars.end(), varname) != this->availableVars.end()) {
            this->inqVars.push_back(varname);
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Variable %s is not in available varialbes", varname.c_str());
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError("No available Vars. Read header first.");
        return;
    }
    // erase non-unique occurances
    std::sort(this->inqVars.begin(), this->inqVars.end());
    auto last = std::unique(this->inqVars.begin(), this->inqVars.end());
    this->inqVars.erase(last, this->inqVars.end());
}

std::vector<std::string> CallADIOSData::getVarsToInquire() { return inqVars; }

std::vector<std::string> CallADIOSData::getAvailableVars() { return availableVars; }

void CallADIOSData::setAvailableVars(std::vector<std::string> avars) { this->availableVars = avars; }

void CallADIOSData::setData(std::shared_ptr<adiosDataMap> _dta) {
    this->dataptr = _dta; }

std::shared_ptr<abstractContainer> CallADIOSData::getData(std::string _str) { return this->dataptr->at(_str); }

} // end namespace adios
} // end namespace megamol