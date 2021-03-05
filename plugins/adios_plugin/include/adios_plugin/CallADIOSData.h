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
#include "mmcore/utility/log/Log.h"

namespace megamol {
namespace adios {


class abstractContainer {
public:
    virtual ~abstractContainer() = default;
    virtual std::vector<float> GetAsFloat() = 0;
    virtual std::vector<double> GetAsDouble() = 0;
    virtual std::vector<int32_t> GetAsInt32() = 0;
    virtual std::vector<uint64_t> GetAsUInt64() = 0;
    virtual std::vector<uint32_t> GetAsUInt32() = 0;
    virtual std::vector<char> GetAsChar() = 0;
    virtual std::vector<unsigned char> GetAsUChar() = 0;
    virtual std::vector<std::string> GetAsString() = 0;


    virtual const std::string getType() = 0;
    virtual const size_t getTypeSize() = 0;
    virtual size_t size() = 0;
    std::vector<size_t> getShape() {
        if (shape.empty()) {
            std::vector<size_t> size_vec = {size()};
            return size_vec;
        }
        return shape;
    }

    std::vector<size_t> shape;
    bool singleValue = false;
};

class DoubleContainer : public abstractContainer {
    typedef double value_type;

public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "double"; }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }
    
    template<class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value || std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

        template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template<class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<char, R>::value ||
                                     std::is_same<unsigned char, R>::value || std::is_same<std::string, R>::value), R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class FloatContainer : public abstractContainer {
    typedef float value_type;

public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "float"; }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }
    
    template <class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value || std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template <class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<char, R>::value ||
                                     std::is_same<unsigned char, R>::value || std::is_same<std::string, R>::value), R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class Int32Container : public abstractContainer {
    typedef int32_t value_type;

public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<int32_t>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "int32_t"; }
    const size_t getTypeSize() override {
        return sizeof(int32_t);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<int32_t, R>::value, R>> getAs() {
        return dataVec;
    }
    
    template <class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value || std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template <class R>
    std::vector<std::enable_if_t<!(std::is_same<int32_t, R>::value || std::is_same<char, R>::value ||
                                     std::is_same<unsigned char, R>::value || std::is_same<std::string, R>::value), R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<int32_t> dataVec;
};

class UInt64Container : public abstractContainer {
    typedef uint64_t value_type;

public:
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "uint64_t"; }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value || std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template <class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<char, R>::value ||
                                     std::is_same<unsigned char, R>::value || std::is_same<std::string, R>::value), R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class UInt32Container : public abstractContainer {
    typedef uint32_t value_type;

public:
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "uint32_t"; }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value || std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template <class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<char, R>::value ||
                                     std::is_same<unsigned char, R>::value || std::is_same<std::string, R>::value),
        R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class UCharContainer : public abstractContainer {
    typedef unsigned char value_type;
    public:
    std::vector<float> GetAsFloat() override { return this->getAs<float>(); }
    std::vector<double> GetAsDouble() override { return this->getAs<double>(); }
    std::vector<int32_t> GetAsInt32() override { return this->getAs<int32_t>(); }
    std::vector<uint64_t> GetAsUInt64() override { return this->getAs<uint64_t>(); }
    std::vector<uint32_t> GetAsUInt32() override { return this->getAs<uint32_t>(); }
    std::vector<char> GetAsChar() override { return this->getAs<char>(); }
    std::vector<unsigned char> GetAsUChar() override { return this->getAs<unsigned char>(); }
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override { return dataVec.size(); }
    const std::string getType() override { return "unsigned char"; }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }

    template <class R> std::vector<std::enable_if_t<(std::is_same<char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template <class R>
    std::vector<std::enable_if_t<
        !(std::is_same<value_type, R>::value || std::is_same<char, R>::value || std::is_same<std::string, R>::value), R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class CharContainer : public abstractContainer {
    typedef char value_type;

public:
    std::vector<float> GetAsFloat() override {
        return this->getAs<float>();
    }
    std::vector<double> GetAsDouble() override {
        return this->getAs<double>();
    }
    std::vector<int32_t> GetAsInt32() override {
        return this->getAs<int32_t>();
    }
    std::vector<uint64_t> GetAsUInt64() override {
        return this->getAs<uint64_t>();
    }
    std::vector<uint32_t> GetAsUInt32() override {
        return this->getAs<uint32_t>();
    }
    std::vector<char> GetAsChar() override {
        return this->getAs<char>();
    }
    std::vector<unsigned char> GetAsUChar() override {
        return this->getAs<unsigned char>();
    }
    std::vector<std::string> GetAsString() override {
        return this->getAs<std::string>();
    }

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override {
        return dataVec.size();
    }
    const std::string getType() override {
        return "char";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<unsigned char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<std::string, R>::value), R>> getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template<class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<unsigned char, R>::value ||
                                     std::is_same<std::string, R>::value),
        R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

    std::vector<value_type> dataVec;
};

class StringContainer : public abstractContainer {
    typedef std::string value_type;

public:
    std::vector<float> GetAsFloat() override {return this->getAs<float>();}
    std::vector<double> GetAsDouble() override {return this->getAs<double>();}
    std::vector<int32_t> GetAsInt32() override {return this->getAs<int32_t>();}
    std::vector<uint64_t> GetAsUInt64() override {return this->getAs<uint64_t>();}
    std::vector<uint32_t> GetAsUInt32() override {return this->getAs<uint32_t>();}
    std::vector<char> GetAsChar() override {return this->getAs<char>();}
    std::vector<unsigned char> GetAsUChar() override {return this->getAs<unsigned char>();}
    std::vector<std::string> GetAsString() override {return this->getAs<std::string>();}

    std::vector<value_type>& getVec() {
        return dataVec;
    }
    size_t size() override {return dataVec.size();}
    const std::string getType() override {return "string";}
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }

private:
    // TODO: maybe better in abstract container - no copy paste
    template<class R>
    std::vector<std::enable_if_t<std::is_same<value_type, R>::value, R>> getAs() {
        return dataVec;
    }

    template<class R>
    std::vector<std::enable_if_t<(std::is_same<char, R>::value), R>> getAs() {
        return reinterpret_cast<std::vector<R>&>(dataVec);
    }

    template<class R>
    std::vector<std::enable_if_t<!(std::is_same<value_type, R>::value || std::is_same<char, R>::value), R>> getAs() {
        throw std::exception("[CallADIOSData] Conversion not supported.");
    }

    std::vector<value_type> dataVec;
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
    static uint32_t FunctionCount(void) { return 2; }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(uint32_t idx) {
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

    bool inquire(const std::string& varname);

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
