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
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {


class abstractContainer {
public:
    virtual ~abstractContainer() = default;
    virtual std::vector<float> GetAsFloat() = 0;
    virtual std::vector<double> GetAsDouble() = 0;
    virtual std::vector<int> GetAsInt() = 0;
    virtual std::vector<unsigned long long int> GetAsUInt64() = 0;
    virtual std::vector<unsigned int> GetAsUInt32() = 0;
    virtual std::vector<char> GetAsChar() = 0;
    virtual std::vector<unsigned char> GetAsUChar() = 0;


    virtual const std::string getType() = 0;
    virtual const size_t getTypeSize() = 0;
    virtual size_t size() = 0;
};

class DoubleContainer : public abstractContainer {
public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<double>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "double"; }
    const size_t getTypeSize() override { return sizeof(double); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<double, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<double, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    std::vector<double> dataVec;
};

class FloatContainer : public abstractContainer {
public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<float>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "float"; }
    const size_t getTypeSize() override { return sizeof(float); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<float, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<float, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

public:
    ~FloatContainer() override = default;

private:
    std::vector<float> dataVec;
};

class IntContainer : public abstractContainer {
public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<int>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "int"; }
    const size_t getTypeSize() override { return sizeof(int); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<int, R>::value, R>> getAs() { return dataVec; }

    template <class R> std::vector<std::enable_if_t<!std::is_same<int, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    std::vector<int> dataVec;
};

class UInt64Container : public abstractContainer {
public:
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<unsigned long long int>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "unsigned long long int"; }
    const size_t getTypeSize() override { return sizeof(unsigned long long int); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<unsigned long long int, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R> std::vector<std::enable_if_t<!std::is_same<unsigned long long int, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    std::vector<unsigned long long int> dataVec;
};

class UInt32Container : public abstractContainer {
public:
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<unsigned int>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "unsigned int"; }
    const size_t getTypeSize() override { return sizeof(unsigned int); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<unsigned int, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R> std::vector<std::enable_if_t<!std::is_same<unsigned int, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    std::vector<unsigned int> dataVec;
};

class UCharContainer : public abstractContainer {
public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int> GetAsInt() override { return this->getAs<int>(); }
    std::vector<unsigned long long int> GetAsUInt64() override { return this->getAs<unsigned long long int>(); }
    std::vector<unsigned int> GetAsUInt32() override { return this->getAs<unsigned int>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }

    std::vector<unsigned char>& getVec() { return dataVec; }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "unsigned char"; }
    const size_t getTypeSize() override { return sizeof(unsigned char); }

private:
    // TODO: maybe better in abstract container - no copy paste
    template <class R> std::vector<std::enable_if_t<std::is_same<unsigned char, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R> std::vector<std::enable_if_t<!std::is_same<unsigned char, R>::value, R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    std::vector<unsigned char> dataVec;
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
            return nullptr;
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
    float getTime() const;

    void inquire(const std::string& varname);

    std::vector<std::string> getVarsToInquire() const;

    std::vector<std::string> getAvailableVars() const;
    void setAvailableVars(const std::vector<std::string>& avars);

    void setDataHash(size_t datah) { this->dataHash = datah; }
    size_t getDataHash() const { return this->dataHash; }

    void setFrameCount(size_t fcount) { this->frameCount = fcount; }
    size_t getFrameCount() const { return this->frameCount; }

    void setFrameIDtoLoad(size_t fid) { this->frameIDtoLoad = fid; }
    size_t getFrameIDtoLoad() const { return this->frameIDtoLoad; }

    void setData(std::shared_ptr<adiosDataMap> _dta);
    std::shared_ptr<abstractContainer> getData(std::string _str) const;

    bool isInVars(std::string);

private:
    size_t dataHash;
    float time;
    size_t frameCount;
    size_t frameIDtoLoad;
    std::vector<std::string> inqVars;
    std::vector<std::string> availableVars;

    std::shared_ptr<adiosDataMap> dataptr;
};

typedef core::factories::CallAutoDescription<CallADIOSData> CallADIOSDataDescription;

} // end namespace adios
} // end namespace megamol