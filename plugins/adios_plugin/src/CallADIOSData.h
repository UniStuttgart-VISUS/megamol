/*
 * CallADIOSData.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace adios {



class abstractContainer {
public:
    virtual const std::vector<float> GetAsFloat() = 0;
    virtual const std::vector<double> GetAsDouble() = 0;
    virtual const std::vector<int> GetAsInt() = 0;
};

class DoubleContainer : public abstractContainer {
public:
    const std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    const std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    const std::vector<int> GetAsInt() override { return this->getAs<int>(); }

    std::vector<double>& getVec() { return dataVec; }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<double, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<double, R>::value, R>> getAs() {
        std::vector<R> convert(dataVec.begin(), dataVec.end());
        return convert;
    }

    std::vector<double> dataVec;
};

class FloatContainer : public abstractContainer {
public:
    const std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    const std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    const std::vector<int> GetAsInt() override { return this->getAs<int>(); }

    std::vector<float>& getVec() { return dataVec; }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<float, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<float, R>::value, R>> getAs() {
        std::vector<R> convert(dataVec.begin(), dataVec.end());
        return convert;
    }

    std::vector<float> dataVec;
};

class IntContainer : public abstractContainer {
public:
    const std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    const std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    const std::vector<int> GetAsInt() override { return this->getAs<int>(); }

    std::vector<int>& getVec() { return dataVec; }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<int, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<int, R>::value, R>> getAs() {
        std::vector<R> convert(dataVec.begin(), dataVec.end());
        return convert;
    }

    std::vector<int> dataVec;
};

typedef std::map<std::string, std::shared_ptr<abstractContainer>> adiosDataMap;

class CallADIOSData : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallADIOSData"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call for ADIOS data"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 2; }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetHeader";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallADIOSData();

    /** Dtor. */
    virtual ~CallADIOSData(void);

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    // CallADIOSData& operator=(const CallADIOSData& rhs); // for now use the default copy constructor


    void setTime(float time);
    float getTime();

    void inquire(std::string varname);

    std::vector<std::string> getVarsToInquire();

    std::vector<std::string> getAvailableVars();
    void setAvailableVars(std::vector<std::string> avar);

    void setDataHash(size_t datah) { this->dataHash = datah; }
    size_t getDataHash() { return this->dataHash; }

    void setFrameCount(size_t fcount) { this->frameCount = fcount; }
    size_t getFrameCount() { return this->frameCount; }

    void setFrameIDtoLoad(size_t fid) { this->frameIDtoLoad = fid; }
    size_t getFrameIDtoLoad() { return this->frameIDtoLoad; }

    void setData(std::shared_ptr<adiosDataMap> _dta);
    std::shared_ptr<abstractContainer> getData(std::string _str);

private:
    size_t dataHash;
    float time;
    size_t frameCount;
    size_t frameIDtoLoad = 0;
    std::vector<std::string> inqVars;
    std::vector<std::string> availableVars;

    std::shared_ptr<adiosDataMap> dataptr;
};

typedef core::factories::CallAutoDescription<CallADIOSData> CallADIOSDataDescription;

} // end namespace adios
} // end namespace megamol