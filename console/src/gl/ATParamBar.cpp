/*
 * gl/ATParamBar.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/ATParamBar.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include <algorithm>
#include <cstdint>
#include "vislib/sys/KeyCode.h"
#include <cassert>
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::console;

gl::ATParamBar::ATParamBar(void *hCore) : ATBar("paramBar"), hCore(hCore), root(),
    lastParamHash(0) {
    std::stringstream def;
    def << Name() << " "
        << "label='Parameters' "
        << "help='ParamSlots published by all instantiated modules' "
        << "color='200 220 240' alpha='192' text='dark' "
        << "position='16 16' "
        << "size='256 512' valueswidth='128' ";
    ::TwDefine(def.str().c_str());
    root.name.clear();
}

gl::ATParamBar::~ATParamBar() {
    hCore = nullptr; // we don't own this memory, therefore we don't delete it
}

void gl::ATParamBar::Update() {
    auto currentParamHash = mmcGetGlobalParameterHash(hCore);
    if (lastParamHash == currentParamHash) return;
    lastParamHash = currentParamHash;
    vislib::sys::Log::DefaultLog.WriteInfo("ATParamBar: Updating TweakBar\n");

    // first update the data representation of the parameter tree
    group enumIn;
    enumIn.name.clear();
    ::mmcEnumParametersA(hCore, (mmcEnumStringAFunction)&mmcCollectParams, &enumIn);
    //if (enumIn.structEqual(&root)) return; // param tree is up to date

    // Recreate param tree
    TwRemoveAllVars(Handle()); // TODO: does this remove groups as well? What about variable types?
    root.params = std::move(enumIn.params);
    root.subgroups = std::move(enumIn.subgroups);
    root.sort();
    unsigned int idCnt = 0;
    root.setIds(idCnt);
    addParamVariables("", &root);
}

gl::ATParamBar::group* gl::ATParamBar::group::findSubGroup(std::string name) const {
    auto e = subgroups.end();
    auto i = std::find_if(subgroups.begin(), e, [&](const std::shared_ptr<group> &o) {
        return o->name == name;
    });
    if (i == e) return nullptr;
    return i->get();
}

gl::ATParamBar::Param* gl::ATParamBar::group::findParam(std::string name) const {
    auto e = params.end();
    auto i = std::find_if(params.begin(), e, [&](const std::shared_ptr<Param> &o) {
        return o->GetName() == name;
    });
    if (i == e) return nullptr;
    return i->get();
}

bool gl::ATParamBar::group::structEqual(group* rhs) const {
    if (name != rhs->name) return false;
    if (subgroups.size() != rhs->subgroups.size()) return false;
    if (params.size() != rhs->params.size()) return false;

    auto gb = rhs->subgroups.begin();
    auto ge = rhs->subgroups.end();
    for (auto g : subgroups) {
        auto gi = std::find_if(gb, ge, [&](const std::shared_ptr<group> &o) { return o->name == g->name; });
        if (gi == ge) return false;
        if (!g->structEqual(gi->get())) return false;
    }

    auto pb = rhs->params.begin();
    auto pe = rhs->params.end();
    for (auto p : params) {
        auto pi = std::find_if(pb, pe, [&](const std::shared_ptr<Param> &o) { return o->GetName() == p->GetName(); });
        if (pi == pe) return false;
    }

    return true;
}

void gl::ATParamBar::group::sort() {
    std::sort(subgroups.begin(), subgroups.end(), [](const std::shared_ptr<group> &lhs, const std::shared_ptr<group> &rhs) { return lhs->name < rhs->name; });
    for (std::shared_ptr<group> g : subgroups) g->sort();
    std::sort(params.begin(), params.end(), [](const std::shared_ptr<Param> &lhs, const std::shared_ptr<Param> &rhs) { return lhs->GetName() < rhs->GetName(); });
}

void gl::ATParamBar::group::setIds(unsigned int &i) {
    id = i++;
    for (std::shared_ptr<group> g : subgroups) g->setIds(i);
    for (std::shared_ptr<Param> p : params) p->SetID(i++);
}

void MEGAMOLCORE_CALLBACK gl::ATParamBar::mmcCollectParams(const char *str, gl::ATParamBar::group *data) {
    vislib::Array<vislib::StringA> nameAry = vislib::StringTokeniserA::Split(str, "::", true);
    std::vector<std::string> name;
    name.resize(nameAry.Count());
    for (size_t i = 0; i < nameAry.Count(); ++i) name[i] = nameAry[i].PeekBuffer();
    nameAry.Clear();

    while (name.size() > 1) {
        group *g = data->findSubGroup(name[0].c_str());
        if (g != nullptr) {
            data = g;
            name.erase(name.begin());
            continue;
        }
        Param *p = data->findParam(name[0].c_str());
        if (p != nullptr) return; // naming conflict!

        // group does not exist yet
        data->subgroups.push_back(std::make_shared<group>());
        data = data->subgroups.back().get();
        data->name = name[0];
        name.erase(name.begin());
    }

    group *g = data->findSubGroup(name[0].c_str());
    if (g != nullptr) return; // naming conflict!
    Param *p = data->findParam(name[0].c_str());
    if (p != nullptr) return; // naming conflict!

    // param does not exist yet
    data->params.push_back(std::make_shared<Param>(std::string(name[0].c_str())));
}

void gl::ATParamBar::addParamVariables(std::string name, group *grp) {
    std::stringstream def;

    for (std::shared_ptr<Param>& p : grp->params) {
        addParamVariable(p, name + "::", p->GetName(), grp->id);
    }

    for (std::shared_ptr<group> g : grp->subgroups) {
        addParamVariables(name + "::" + g->name, g.get());

        if (grp->id > 0) {
            def.str("");
            def << Name() << "/G" << std::to_string(g->id) << " "
                << "group='G" << std::to_string(grp->id) << "' ";
            ::TwDefine(def.str().c_str());
        }
    }

    if (grp->id > 0) {
        def.str("");
        def << Name() << "/G" << std::to_string(grp->id) << " "
            << "label='" << grp->name << "' ";
        ::TwDefine(def.str().c_str());
    }
}

void gl::ATParamBar::addParamVariable(std::shared_ptr<Param>& param, std::string name_prefix, std::string name, unsigned int grpId) {
    std::stringstream def;
    def << "label='" << name << "' ";
    if (grpId > 0) def << "group='" << "G" << std::to_string(grpId) << "' ";

    ::mmcGetParameterA(hCore, (name_prefix + name).c_str(), param->Handle());
    if (!param->Handle().IsValid()) return; // failed to get param description
    unsigned int descLen = 0;
    ::mmcGetParameterTypeDescription(param->Handle(), nullptr, &descLen);
    std::vector<unsigned char> desc(descLen);
    ::mmcGetParameterTypeDescription(param->Handle(), desc.data(), &descLen);
    if (descLen != desc.size()) desc.resize(descLen);
    if (descLen < 6) return; // invalid description data

    std::shared_ptr<Param> oldParam = param;
    if (::memcmp(desc.data(), "MMBUTN", 6) == 0) param = std::make_shared<ButtonParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMBOOL", 6) == 0) param = std::make_shared<BoolParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMENUM", 6) == 0) param = std::make_shared<EnumParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMFENU", 6) == 0) param = std::make_shared<FlexEnumParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMFLOT", 6) == 0) param = std::make_shared<FloatParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMINTR", 6) == 0) param = std::make_shared<IntParam>(oldParam, Handle(), def, desc);
    else if (::memcmp(desc.data(), "MMVC3F", 6) == 0) param = std::make_shared<Vec3fParam>(oldParam, Handle(), def, desc);
    else param = std::make_shared<StringParam>(oldParam, Handle(), def, desc);
}

/****************************************************************************/

gl::ATParamBar::Param::Param(const std::string& name) : id(0), name(name), hParam() {
    // intentionally empty
}

gl::ATParamBar::Param::~Param() {
    // intentionally empty
}

gl::ATParamBar::Param::Param(Param && src) : id(src.id), name(std::move(src.name)), hParam(std::move(src.hParam)) {
    src.id = 0;
    assert(src.name.empty());
    assert(!src.hParam.IsValid());
}

/****************************************************************************/

gl::ATParamBar::ButtonParam::ButtonParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : Param(std::move(*src.get())) {
    uint16_t keyCode = 0;
    if (desc.size() == 7) keyCode = *reinterpret_cast<const char*>(desc.data() + 6);
    else if (desc.size() == 8) keyCode = *reinterpret_cast<const uint16_t*>(desc.data() + 6);
    vislib::sys::KeyCode key(keyCode);

    if (key.NoModKeys() != 0) {
        vislib::StringA keyStr;
        if (key.IsSpecial()) {
            switch (key.NoModKeys()) {
            case vislib::sys::KeyCode::KEY_ENTER: keyStr = "RET"; break;
            case vislib::sys::KeyCode::KEY_ESC: keyStr = "ESC"; break;
            case vislib::sys::KeyCode::KEY_TAB: keyStr = "TAB"; break;
            case vislib::sys::KeyCode::KEY_LEFT: keyStr = "LEFT"; break;
            case vislib::sys::KeyCode::KEY_UP: keyStr = "UP"; break;
            case vislib::sys::KeyCode::KEY_RIGHT: keyStr = "RIGHT"; break;
            case vislib::sys::KeyCode::KEY_DOWN: keyStr = "DOWN"; break;
            case vislib::sys::KeyCode::KEY_PAGE_UP: keyStr = "PGUP"; break;
            case vislib::sys::KeyCode::KEY_PAGE_DOWN: keyStr = "PGDOWN"; break;
            case vislib::sys::KeyCode::KEY_HOME: keyStr = "HOME"; break;
            case vislib::sys::KeyCode::KEY_END: keyStr = "END"; break;
            case vislib::sys::KeyCode::KEY_INSERT: keyStr = "INS"; break;
            case vislib::sys::KeyCode::KEY_DELETE: keyStr = "DEL"; break;
            case vislib::sys::KeyCode::KEY_BACKSPACE: keyStr = "BS"; break;
            case vislib::sys::KeyCode::KEY_F1: keyStr = "F1"; break;
            case vislib::sys::KeyCode::KEY_F2: keyStr = "F2"; break;
            case vislib::sys::KeyCode::KEY_F3: keyStr = "F3"; break;
            case vislib::sys::KeyCode::KEY_F4: keyStr = "F4"; break;
            case vislib::sys::KeyCode::KEY_F5: keyStr = "F5"; break;
            case vislib::sys::KeyCode::KEY_F6: keyStr = "F6"; break;
            case vislib::sys::KeyCode::KEY_F7: keyStr = "F7"; break;
            case vislib::sys::KeyCode::KEY_F8: keyStr = "F8"; break;
            case vislib::sys::KeyCode::KEY_F9: keyStr = "F9"; break;
            case vislib::sys::KeyCode::KEY_F10: keyStr = "F10"; break;
            case vislib::sys::KeyCode::KEY_F11: keyStr = "F11"; break;
            case vislib::sys::KeyCode::KEY_F12: keyStr = "F12"; break;
            }
        } else if (key.NoModKeys() == ' ') {
            keyStr = "SPACE";
        } else {
            keyStr.Format("%c", static_cast<char>(key.NoModKeys()));
        }
        if (!keyStr.IsEmpty()) {
            if (key.IsAltMod()) {
                keyStr.Prepend("ALT+");
            }
            if (key.IsCtrlMod()) {
                keyStr.Prepend("CTRL+");
            }
            if (key.IsShiftMod()) {
                keyStr.Prepend("SHIFT+");
            }
        }

        if (!keyStr.IsEmpty()) def << " key='" << keyStr.PeekBuffer() << "' ";
    }

    ::TwAddButton(bar, GetKey().c_str(), reinterpret_cast<TwButtonCallback>(&ButtonParam::twCallback), this, def.str().c_str());

}

void TW_CALL gl::ATParamBar::ButtonParam::twCallback(ButtonParam *prm) {
    ::mmcSetParameterValue(prm->Handle(), _T("click"));
}

/****************************************************************************/

gl::ATParamBar::ValueParam::ValueParam(std::shared_ptr<Param> src, TwBar* bar, TwType type, std::stringstream& def) : Param(std::move(*src.get())) {
    ::TwAddVarCB(bar, GetKey().c_str(), type, 
        reinterpret_cast<TwSetVarCallback>(&ValueParam::twSetCallback),
        reinterpret_cast<TwGetVarCallback>(&ValueParam::twGetCallback),
        this, def.str().c_str());
}

void TW_CALL gl::ATParamBar::ValueParam::twSetCallback(const void *value, ValueParam *clientData) {
    clientData->Set(value);
}

void TW_CALL gl::ATParamBar::ValueParam::twGetCallback(void *value, ValueParam *clientData) {
    clientData->Get(value);
}

/****************************************************************************/

gl::ATParamBar::StringParam::StringParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : ValueParam(src, bar, TW_TYPE_CDSTRING, def) {
    // intentionally empty
}

void gl::ATParamBar::StringParam::Set(const void *value) {
    ::mmcSetParameterValueA(this->Handle(), *reinterpret_cast<const char * const *>(value));
}

void gl::ATParamBar::StringParam::Get(void *value) {
    const char *dat = ::mmcGetParameterValueA(this->Handle());
    char **destPtr = (char **)value;
    ::TwCopyCDStringToLibrary(destPtr, dat);
}

/****************************************************************************/

gl::ATParamBar::BoolParam::BoolParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : ValueParam(src, bar, TW_TYPE_BOOLCPP, def) {
    // intentionally empty
}

void gl::ATParamBar::BoolParam::Set(const void *value) {
    ::mmcSetParameterValueA(this->Handle(), (*static_cast<const bool*>(value)) ? "true" : "false");
}

void gl::ATParamBar::BoolParam::Get(void *value) {
    try {
        *static_cast<bool*>(value) = vislib::CharTraitsA::ParseBool(::mmcGetParameterValueA(this->Handle()));
    } catch (...) {
    }
}

/****************************************************************************/

gl::ATParamBar::EnumParam::EnumParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc)
        : ValueParam(src, bar, makeMyEnumType(src->Handle(), desc), def), values() {
    parseEnumDesc(values, desc);
}

void gl::ATParamBar::EnumParam::Set(const void *value) {
    vislib::StringA str;
    str.Format("%d", *static_cast<const int*>(value));
    ::mmcSetParameterValueA(this->Handle(), str);
}

void gl::ATParamBar::EnumParam::Get(void *value) {
    try {
        const char *v = ::mmcGetParameterValueA(this->Handle());
        for (SIZE_T i = 0; i < this->values.size(); i++) {
            if (strcmp(v, this->values[i].Label) == 0) {
                *static_cast<int*>(value) = this->values[i].Value;
                return;
            }
        }
        *static_cast<int*>(value) = vislib::CharTraitsA::ParseInt(v);
    } catch (...) {
    }
}

TwType gl::ATParamBar::EnumParam::makeMyEnumType(void* hParam, const std::vector<unsigned char>& desc) {
    vislib::StringA n;
    UINT64 id = reinterpret_cast<UINT64>(hParam);
    unsigned char idc[8];
    ::memcpy(idc, &id, 8);
    n.Format("%.2x%.2x%.2x%.2x%.2x%.2x%.2x%.2x", idc[0], idc[1], idc[2], idc[3], idc[4], idc[5], idc[6], idc[7]);

    std::vector<TwEnumVal> values;
    parseEnumDesc(values, desc);

    return ::TwDefineEnum(n, values.data(), static_cast<unsigned int>(values.size()));
}

void gl::ATParamBar::EnumParam::parseEnumDesc(std::vector<TwEnumVal>& outValues, const std::vector<unsigned char>& desc) {
    int dp = 6;
    SIZE_T cnt = *reinterpret_cast<const unsigned int*>(desc.data() + dp);
    dp += 4;
    outValues.resize(cnt);
    for (SIZE_T i = 0; i < cnt; i++) {
        outValues[i].Value = static_cast<int>(*reinterpret_cast<const unsigned int*>(desc.data() + dp));

        dp += 4;
        vislib::StringA l(reinterpret_cast<const char*>(desc.data() + dp));
        dp += l.Length() + 1;

        auto p = std::find_if(enumStrings.begin(), enumStrings.end(), [&l](std::shared_ptr<vislib::StringA> const& o) {
            return (*o) == l;
        });
        if (p == enumStrings.end()) {
            enumStrings.push_back(std::make_shared<vislib::StringA>(l));
            p = enumStrings.end() - 1;
        }
        outValues[i].Label = (*p)->PeekBuffer();
    }
}

std::vector<std::shared_ptr<vislib::StringA> > gl::ATParamBar::EnumParam::enumStrings;

/****************************************************************************/

gl::ATParamBar::FlexEnumParam::FlexEnumParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc)
    : ValueParam(src, bar, makeMyFlexEnumType(src->Handle(), desc), def) {

    std::vector<TwEnumVal> values;
    parseFlexEnumDesc(values, desc);
    ::TwRemoveVar(bar, GetKey().c_str());
    auto type = ::TwDefineEnum(GetKey().c_str(), values.data(), static_cast<unsigned int>(values.size()));
    ::TwAddVarCB(bar, GetKey().c_str(), type,
        reinterpret_cast<TwSetVarCallback>(&ValueParam::twSetCallback),
        reinterpret_cast<TwGetVarCallback>(&ValueParam::twGetCallback),
        this, def.str().c_str());
}

void gl::ATParamBar::FlexEnumParam::Set(const void *value) {
    auto str = this->enumStrings[*static_cast<const int *>(value)];
    ::mmcSetParameterValueA(this->Handle(), *str);
}

void gl::ATParamBar::FlexEnumParam::Get(void *value) {
    try {
        const char *dat = ::mmcGetParameterValueA(this->Handle());

        //char **destPtr = (char **)value;
        //::TwCopyCDStringToLibrary(destPtr, dat);

        for (SIZE_T i = 0; i < this->enumStrings.size(); i++) {
            if (this->enumStrings[i]->Equals(dat)) {
                *static_cast<int*>(value) = static_cast<int>(i); //this->values[i].Value;
                return;
            }
        }
        *static_cast<int*>(value) = 0; // vislib::CharTraitsA::ParseInt(v);
        if (this->enumStrings.size() > 0) {
            ::mmcSetParameterValueA(this->Handle(), *this->enumStrings[0]);
        } else {
            ::mmcSetParameterValueA(this->Handle(), "<undef>");
        }
    } catch (...) {
    }
}

TwType gl::ATParamBar::FlexEnumParam::makeMyFlexEnumType(void* hParam, const std::vector<unsigned char>& desc) {
    vislib::StringA n;
    UINT64 id = reinterpret_cast<UINT64>(hParam);
    unsigned char idc[8];
    ::memcpy(idc, &id, 8);
    n.Format("%.2x%.2x%.2x%.2x%.2x%.2x%.2x%.2x", idc[0], idc[1], idc[2], idc[3], idc[4], idc[5], idc[6], idc[7]);
    std::vector<TwEnumVal> values;
//    parseFlexEnumDesc(values, desc);
    return ::TwDefineEnum(n, values.data(), static_cast<unsigned int>(values.size()));
}

void gl::ATParamBar::FlexEnumParam::parseFlexEnumDesc(std::vector<TwEnumVal>& outValues, const std::vector<unsigned char>& desc) {
    int dp = 6;
    SIZE_T cnt = *reinterpret_cast<const unsigned int*>(desc.data() + dp);
    dp += 4;
    outValues.resize(cnt);
    for (SIZE_T i = 0; i < cnt; i++) {
        outValues[i].Value = static_cast<int>(i); //static_cast<int>(*reinterpret_cast<const unsigned int*>(desc.data() + dp));

        vislib::StringA l(reinterpret_cast<const char*>(desc.data() + dp));
        dp += l.Length() + 1;

        auto p = std::find_if(enumStrings.begin(), enumStrings.end(), [&l](std::shared_ptr<vislib::StringA> const& o) {
            return (*o) == l;
        });
        if (p == enumStrings.end()) {
            enumStrings.push_back(std::make_shared<vislib::StringA>(l));
            p = enumStrings.end() - 1;
        }
        outValues[i].Label = (*p)->PeekBuffer();
    }
}

//std::vector<std::shared_ptr<vislib::StringA> > gl::ATParamBar::FlexEnumParam::enumStrings;

/****************************************************************************/

gl::ATParamBar::FloatParam::FloatParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : ValueParam(src, bar, TW_TYPE_FLOAT, def) {
    float step = 0.001f;
    if (desc.size() == 14) {
        float minVal = *reinterpret_cast<const float*>(desc.data() + 6);
        float maxVal = *reinterpret_cast<const float*>(desc.data() + 10);
        ::TwSetParam(bar, GetKey().c_str(), "min", TW_PARAM_FLOAT, 1, &minVal);
        ::TwSetParam(bar, GetKey().c_str(), "max", TW_PARAM_FLOAT, 1, &maxVal);
    }
    ::TwSetParam(bar, GetKey().c_str(), "step", TW_PARAM_FLOAT, 1, &step);
}

void gl::ATParamBar::FloatParam::Set(const void *value) {
    vislib::StringA str;
    str.Format("%f", *static_cast<const float*>(value));
    ::mmcSetParameterValueA(this->Handle(), str);
}

void gl::ATParamBar::FloatParam::Get(void *value) {
    try {
        *static_cast<float*>(value) = static_cast<float>(vislib::CharTraitsA::ParseDouble(::mmcGetParameterValueA(this->Handle())));
    } catch (...) {
    }
}

/****************************************************************************/

gl::ATParamBar::IntParam::IntParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : ValueParam(src, bar, TW_TYPE_INT32, def) {
    if (desc.size() == 6 + 2 * sizeof(int)) {
        int minVal = *reinterpret_cast<const int*>(desc.data() + 6);
        int maxVal = *reinterpret_cast<const int*>(desc.data() + 6 + sizeof(int));
        ::TwSetParam(bar, GetKey().c_str(), "min", TW_PARAM_INT32, 1, &minVal);
        ::TwSetParam(bar, GetKey().c_str(), "max", TW_PARAM_INT32, 1, &maxVal);
    }
}

void gl::ATParamBar::IntParam::Set(const void *value) {
    vislib::StringA str;
    str.Format("%d", *static_cast<const int*>(value));
    ::mmcSetParameterValueA(this->Handle(), str);
}

void gl::ATParamBar::IntParam::Get(void *value) {
    try {
        *static_cast<int*>(value) = vislib::CharTraitsA::ParseInt(::mmcGetParameterValueA(this->Handle()));
    } catch (...) {
    }
}

/****************************************************************************/

gl::ATParamBar::Vec3fParam::Vec3fParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc) : ValueParam(src, bar, makeMyStructType(), def) {
    // intentionally empty
}

void gl::ATParamBar::Vec3fParam::Set(const void *value) {
    vislib::StringA s;
    const float* f = static_cast<const float*>(value);
    s.Format("%f;%f;%f", f[0], f[1], f[2]);
    ::mmcSetParameterValueA(this->Handle(), s.PeekBuffer());
}

void gl::ATParamBar::Vec3fParam::Get(void *value) {
    float f[3];
    const char *dat = ::mmcGetParameterValueA(this->Handle());
#ifdef _WIN32
    sscanf_s
#else
    sscanf
#endif
        (dat, "%f;%f;%f", &f[0], &f[1], &f[2]);
    ::memcpy(value, f, 3 * sizeof(float));
}

TwType gl::ATParamBar::Vec3fParam::makeMyStructType() {
    static TwType t = TW_TYPE_UNDEF;
    if (t == TW_TYPE_UNDEF) {
        const char* MY_ATB_FLOAT_STEPS = "step = '0.001'";
        TwStructMember vec3fmembers[] = {
            { "x", TW_TYPE_FLOAT, 0, MY_ATB_FLOAT_STEPS },
            { "y", TW_TYPE_FLOAT, sizeof(float), MY_ATB_FLOAT_STEPS },
            { "z", TW_TYPE_FLOAT, 2 * sizeof(float), MY_ATB_FLOAT_STEPS }
        };

        t = TwDefineStruct("vec3f", vec3fmembers, 3, 3 * sizeof(float), &Vec3fParam::mySummaryCallback, nullptr);
    }
    return t;
}

void TW_CALL gl::ATParamBar::Vec3fParam::mySummaryCallback(char *summaryString, size_t summaryMaxLength, const void *value, void *summaryClientData) {
    vislib::StringA s;
    const float* f = static_cast<const float*>(value);
    s.Format("%f;%f;%f", f[0], f[1], f[2]);
    memcpy(summaryString, s.PeekBuffer(), std::max<size_t>(summaryMaxLength, s.Length()));
}

#endif
