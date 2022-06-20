/*
 * EnumParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/param/EnumParam.h"
#include "vislib/IllegalStateException.h"
#include "vislib/StringConverter.h"
#include "vislib/UTF8Encoder.h"

using namespace megamol::core::param;


/*
 * EnumParam::EnumParam
 */
EnumParam::EnumParam(int initVal) : AbstractParam(), val(), typepairs() {
    this->InitPresentation(AbstractParamPresentation::ParamType::ENUM);
    this->SetValue(initVal);
}


/*
 * EnumParam::~EnumParam
 */
EnumParam::~EnumParam(void) {
    // intentionally empty
}


/*
 * EnumParam::ClearTypePairs
 */
void megamol::core::param::EnumParam::ClearTypePairs(void) {
    if (this->isSlotPublic()) {
        throw vislib::IllegalStateException(
            "You must not modify an enum parameter which is already public", __FILE__, __LINE__);
    }
    this->typepairs.Clear();
}


/*
 * EnumParam::Definition
 */
std::string EnumParam::Definition() const {
    vislib::StringA utf8;
    unsigned int s = 6;
    unsigned int c = 0;
    vislib::ConstIterator<vislib::Map<int, vislib::TString>::Iterator> constIter = this->typepairs.GetConstIterator();
    while (constIter.HasNext()) {
        const vislib::Map<int, vislib::TString>::ElementPair& pair = constIter.Next();
        s += sizeof(int) + vislib::UTF8Encoder::CalcUTF8Size(pair.Value());
    }
    s += sizeof(unsigned int);

    vislib::RawStorage outDef;
    outDef.AssertSize(s);
    memcpy(outDef.AsAt<char>(0), "MMENUM", 6);
    s = 6 + sizeof(unsigned int);
    constIter = this->typepairs.GetConstIterator();
    while (constIter.HasNext()) {
        const vislib::Map<int, vislib::TString>::ElementPair& pair = constIter.Next();
        if (!vislib::UTF8Encoder::Encode(utf8, pair.Value())) {
            continue;
        }
        unsigned int utf8size = utf8.Length() + 1;

        *outDef.AsAt<int>(s) = pair.Key();
        s += sizeof(int);
        memcpy(outDef.AsAt<char>(s), utf8.PeekBuffer(), utf8size);
        s += utf8size;
        c++;
    }
    *outDef.AsAt<unsigned int>(6) = c;

    std::string return_str;
    return_str.resize(outDef.GetSize());
    std::copy(outDef.AsAt<char>(0), outDef.AsAt<char>(0) + outDef.GetSize(), return_str.begin());
    return return_str;
}


/*
 * EnumParam::ParseValue
 */
bool EnumParam::ParseValue(std::string const& v) {
    try {
        vislib::SingleLinkedList<int> keys = this->typepairs.FindKeys(v.c_str());
        if (keys.IsEmpty()) {
            this->SetValue(vislib::TCharTraits::ParseInt(v.c_str()));
        } else {
            this->SetValue(keys.First());
        }
        return true;
    } catch (...) {}
    return false;
}


/*
 * EnumParam::SetTypePair
 */
EnumParam* EnumParam::SetTypePair(int value, const char* name) {
    if (this->isSlotPublic()) {
        throw vislib::IllegalStateException(
            "You must not modify an enum parameter which is already public", __FILE__, __LINE__);
    }
    this->typepairs[value] = A2T(name);
    return this;
}


/*
 * EnumParam::SetTypePair
 */
EnumParam* EnumParam::SetTypePair(int value, const wchar_t* name) {
    if (this->isSlotPublic()) {
        throw vislib::IllegalStateException(
            "You must not modify an enum parameter which is already public", __FILE__, __LINE__);
    }
    this->typepairs[value] = W2T(name);
    return this;
}


/*
 * EnumParam::SetValue
 */
void EnumParam::SetValue(int v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty)
            this->setDirty();
    }
}


/*
 * EnumParam::ValueString
 */
std::string EnumParam::ValueString(void) const {
    const vislib::TString* v = this->typepairs.FindValue(this->val);
    if (v != NULL)
        return (*v).PeekBuffer();
    vislib::TString str;
    str.Format(_T("%d"), this->val);
    return str.PeekBuffer();
}
