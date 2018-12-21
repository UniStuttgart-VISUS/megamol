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
    
CallADIOSData::CallADIOSData()
    : dataHash(0)
    , time(0)
    , frameCount(0)
    , dataptr(nullptr) {
}


/**
 * \brief 
 */
CallADIOSData::~CallADIOSData(void) = default;


/**
 * \brief CallADIOSData::setTime
 * \param time 
 */
void CallADIOSData::setTime(float time) { this->time = time; }


/**
 * \brief 
 * \return 
 */
float CallADIOSData::getTime() const { return this->time; }


/**
 * \brief 
 * \param varname 
 */
void CallADIOSData::inquire(const std::string &varname) {
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
    // erase non-unique occurrences
    std::sort(this->inqVars.begin(), this->inqVars.end());
    const auto last = std::unique(this->inqVars.begin(), this->inqVars.end());
    this->inqVars.erase(last, this->inqVars.end());
}

std::vector<std::string> CallADIOSData::getVarsToInquire() const { return inqVars; }

std::vector<std::string> CallADIOSData::getAvailableVars() const { return availableVars; }

void CallADIOSData::setAvailableVars(const std::vector<std::string> &avars) { this->availableVars = avars; }

void CallADIOSData::setData(std::shared_ptr<adiosDataMap> _dta) {
    this->dataptr = _dta; }

std::shared_ptr<abstractContainer> CallADIOSData::getData(std::string _str) const { return this->dataptr->at(_str); }

bool CallADIOSData::isInVars(std::string var) {
    return std::find(this->availableVars.begin(), this->availableVars.end(), var) != this->availableVars.end();
}

} // end namespace adios
} // end namespace megamol