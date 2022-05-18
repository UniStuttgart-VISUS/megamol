/*
 * CallADIOSData.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/utility/log/Log.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

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

template<typename value_type>
class containerInterface {
public:
    std::vector<value_type>& getVec() {
        return dataVec;
    }

protected:
    size_t getSize() {
        return dataVec.size();
    }

    template<class R>
    std::vector<std::enable_if_t<std::is_same_v<value_type, R>, R>> getAs() {
        return dataVec;
    }

    template<class R>
    std::vector<std::enable_if_t<std::is_same_v<std::string, R> && !std::is_same_v<std::string, value_type>, R>>
    getAs() {
        std::vector<R> new_vec(dataVec.size());
        std::transform(
            dataVec.begin(), dataVec.end(), new_vec.begin(), [](const value_type& val) { return std::to_string(val); });
        return new_vec;
    }

    template<class R>
    std::vector<std::enable_if_t<!(std::is_same_v<value_type, R> || std::is_same_v<std::string, R> ||
                                     std::is_same_v<std::string, value_type>),
        R>>
    getAs() {
        std::vector<R> new_vec(dataVec.begin(), dataVec.end());
        return new_vec;
    }

private:
    std::vector<value_type> dataVec;
};

class DoubleContainer : public abstractContainer, public containerInterface<double> {
    typedef double value_type;

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

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "double";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class FloatContainer : public abstractContainer, public containerInterface<float> {
    typedef float value_type;

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

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "float";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class Int32Container : public abstractContainer, public containerInterface<int32_t> {
    typedef int32_t value_type;

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

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "int32_t";
    }
    const size_t getTypeSize() override {
        return sizeof(int32_t);
    }
};

class UInt64Container : public abstractContainer, public containerInterface<uint64_t> {
    typedef uint64_t value_type;

public:
    std::vector<double> GetAsDouble() override {
        return this->getAs<double>();
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
    std::vector<int32_t> GetAsInt32() override {
        return this->getAs<int32_t>();
    }
    std::vector<float> GetAsFloat() override {
        return this->getAs<float>();
    }
    std::vector<unsigned char> GetAsUChar() override {
        return this->getAs<unsigned char>();
    }
    std::vector<std::string> GetAsString() override {
        return this->getAs<std::string>();
    }

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "uint64_t";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class UInt32Container : public abstractContainer, public containerInterface<uint32_t> {
    typedef uint32_t value_type;

public:
    std::vector<double> GetAsDouble() override {
        return this->getAs<double>();
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
    std::vector<int32_t> GetAsInt32() override {
        return this->getAs<int32_t>();
    }
    std::vector<float> GetAsFloat() override {
        return this->getAs<float>();
    }
    std::vector<unsigned char> GetAsUChar() override {
        return this->getAs<unsigned char>();
    }
    std::vector<std::string> GetAsString() override {
        return this->getAs<std::string>();
    }

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "uint32_t";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class UCharContainer : public abstractContainer, public containerInterface<unsigned char> {
    typedef unsigned char value_type;

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

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "unsigned char";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class CharContainer : public abstractContainer, public containerInterface<char> {
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

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "char";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};

class StringContainer : public abstractContainer, public containerInterface<std::string> {
    typedef std::string value_type;

    template<typename T>
    static constexpr T str_conv(std::string const& str) {
        if constexpr (std::is_same_v<float, T>) {
            return std::stof(str);
        } else if constexpr (std::is_same_v<double, T>) {
            return std::stod(str);
        } else if constexpr (std::is_same_v<int32_t, T>) {
            return std::stoi(str);
        } else if constexpr (std::is_same_v<uint64_t, T>) {
            return std::stoull(str);
        } else if constexpr (std::is_same_v<uint32_t, T>) {
            return std::stoul(str);
        } else if constexpr (std::is_same_v<char, T>) {
            return std::stoi(str);
        } else if constexpr (std::is_same_v<unsigned char, T>) {
            return std::stoul(str);
        } else {
            static_assert("Unsupported type");
        }
    }

    template<typename T>
    std::vector<T> get_from_str() {
        auto const& old_vec = getAs<std::string>();
        std::vector<T> new_vec(old_vec.size());
        std::transform(
            old_vec.begin(), old_vec.end(), new_vec.begin(), [](std::string const& str) { return str_conv<T>(str); });
        return new_vec;
    }

public:
    std::vector<float> GetAsFloat() override {
        return get_from_str<float>();
    }
    std::vector<double> GetAsDouble() override {
        return get_from_str<double>();
    }
    std::vector<int32_t> GetAsInt32() override {
        return get_from_str<int32_t>();
    }
    std::vector<uint64_t> GetAsUInt64() override {
        return get_from_str<uint64_t>();
    }
    std::vector<uint32_t> GetAsUInt32() override {
        return get_from_str<uint32_t>();
    }
    std::vector<char> GetAsChar() override {
        return get_from_str<char>();
    }
    std::vector<unsigned char> GetAsUChar() override {
        return get_from_str<unsigned char>();
    }
    std::vector<std::string> GetAsString() override {
        return this->getAs<std::string>();
    }

    size_t size() override {
        return getSize();
    }
    const std::string getType() override {
        return "string";
    }
    const size_t getTypeSize() override {
        return sizeof(value_type);
    }
};


typedef std::map<std::string, std::shared_ptr<abstractContainer>> adiosDataMap;

class CallADIOSData : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallADIOSData";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call for ADIOS data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static uint32_t FunctionCount(void) {
        return 2;
    }

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

    bool inquireVar(const std::string& varname);
    std::vector<std::string> getVarsToInquire() const;
    std::vector<std::string> getAvailableVars() const;
    void setAvailableVars(const std::vector<std::string>& avars);

    bool inquireAttr(const std::string& attrname);
    std::vector<std::string> getAttributesToInquire() const;
    std::vector<std::string> getAvailableAttributes() const;
    void setAvailableAttributes(const std::vector<std::string>& availattribs);

    void setDataHash(size_t datah) {
        this->dataHash = datah;
    }
    size_t getDataHash() const {
        return this->dataHash;
    }

    void setFrameCount(size_t fcount) {
        this->frameCount = fcount;
    }
    size_t getFrameCount() const {
        return this->frameCount;
    }

    void setFrameIDtoLoad(size_t fid) {
        this->frameIDtoLoad = fid;
    }
    size_t getFrameIDtoLoad() const {
        return this->frameIDtoLoad;
    }

    void setData(std::shared_ptr<adiosDataMap> _dta);
    std::shared_ptr<abstractContainer> getData(std::string _str) const;

    bool isInVars(std::string);
    bool isInAttributes(std::string);

private:
    size_t dataHash;
    float time;
    size_t frameCount;
    size_t frameIDtoLoad;
    std::vector<std::string> inqVars;
    std::vector<std::string> availableVars;
    std::vector<std::string> inqAttributes;
    std::vector<std::string> availableAttributes;

    std::shared_ptr<adiosDataMap> dataptr;
};

typedef core::factories::CallAutoDescription<CallADIOSData> CallADIOSDataDescription;

} // end namespace adios
} // end namespace megamol
