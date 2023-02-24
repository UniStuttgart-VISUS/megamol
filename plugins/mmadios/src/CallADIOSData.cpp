/*
 * CallADIOSData.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmadios/CallADIOSData.h"
#include "mmcore/utility/log/Log.h"
#include <algorithm>

namespace megamol::adios {

CallADIOSData::CallADIOSData() : dataHash(0), time(0), frameCount(0), dataptr(nullptr) {}


/**
 * \brief
 */
CallADIOSData::~CallADIOSData(void) = default;


/**
 * \brief CallADIOSData::setTime
 * \param time
 */
void CallADIOSData::setTime(float time) {
    this->time = time;
}


/**
 * \brief
 * \return
 */
float CallADIOSData::getTime() const {
    return this->time;
}


/**
 * \brief Sets variable to inquire.
 * \param varname: The name of the variable.
 * \return Returns false if variable is not available, true otherwise.
 */
bool CallADIOSData::inquireVar(const std::string& varname) {
    if (!this->availableVars.empty()) {
        if (std::find(this->availableVars.begin(), this->availableVars.end(), varname) != this->availableVars.end()) {
            this->inqVars.push_back(varname);
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[CallADIOSData] Variable %s is not in available variables", varname.c_str());
            return false;
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[CallADIOSData] No available Vars. Read header first.");
        return false;
    }
    // erase non-unique occurrences
    std::sort(this->inqVars.begin(), this->inqVars.end());
    const auto last = std::unique(this->inqVars.begin(), this->inqVars.end());
    this->inqVars.erase(last, this->inqVars.end());
    return true;
}

std::vector<std::string> CallADIOSData::getVarsToInquire() const {
    return inqVars;
}

std::vector<std::string> CallADIOSData::getAvailableVars() const {
    return availableVars;
}

void CallADIOSData::setAllVars(std::map<std::string, std::map<std::string, std::string>> vars) {
    allVars = vars;
}

std::map<std::string, std::string> CallADIOSData::getVarProperties(std::string var) const {
    return allVars.at(var);
}

void CallADIOSData::setAvailableVars(const std::vector<std::string>& avars) {
    this->availableVars = avars;
}

bool CallADIOSData::inquireAttr(const std::string& attrname) {
    if (!this->availableVars.empty()) {
        if (std::find(this->availableAttributes.begin(), this->availableAttributes.end(), attrname) !=
            this->availableAttributes.end()) {
            this->inqAttributes.push_back(attrname);
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[CallADIOSData] Attribute %s is not in available attributes", attrname.c_str());
            return false;
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[CallADIOSData] No available Attributes");
        return false;
    }
    // erase non-unique occurrences
    std::sort(this->inqAttributes.begin(), this->inqAttributes.end());
    const auto last = std::unique(this->inqAttributes.begin(), this->inqAttributes.end());
    this->inqAttributes.erase(last, this->inqAttributes.end());
    return true;
}

std::vector<std::string> CallADIOSData::getAttributesToInquire() const {
    return inqAttributes;
}

std::vector<std::string> CallADIOSData::getAvailableAttributes() const {
    return availableAttributes;
}

void CallADIOSData::setAvailableAttributes(const std::vector<std::string>& availattribs) {
    this->availableAttributes = availattribs;
}

void CallADIOSData::setData(std::shared_ptr<adiosDataMap> _dta) {
    this->dataptr = _dta;
    availableVars.clear();
    availableVars.reserve(dataptr->size());
    for (auto& entry : *dataptr) {
        availableVars.emplace_back(entry.first);
    }
}

std::shared_ptr<abstractContainer> CallADIOSData::getData(std::string _str) const {
    return this->dataptr->at(_str);
}

bool CallADIOSData::isInVars(std::string var) {
    return std::find(this->availableVars.begin(), this->availableVars.end(), var) != this->availableVars.end();
}

bool CallADIOSData::isInAttributes(std::string attr) {
    return std::find(this->availableAttributes.begin(), this->availableAttributes.end(), attr) !=
           this->availableAttributes.end();
}

} // namespace megamol::adios
