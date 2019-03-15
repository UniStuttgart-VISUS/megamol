/*
 * PluginsStateFileGeneratorJob.cpp
 *
 * Copyright (C) 2016 by MegaMol Team; TU Dresden.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "job/PluginsStateFileGeneratorJob.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Path.h"
#include "vislib/xmlUtils.h"
#include "vislib/UTF8Encoder.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include <set>
#include <cassert>

using namespace megamol::core;


/*
 * job::PluginsStateFileGeneratorJob::PluginsStateFileGeneratorJob
 */
job::PluginsStateFileGeneratorJob::PluginsStateFileGeneratorJob()
        : AbstractThreadedJob(), Module(),
        fileNameSlot("filename", "fileNameSlot") {
    fileNameSlot.SetParameter(new param::FilePathParam(
        "MegaMolConf."
#if defined _WIN64 || defined _LIN64
        "64"
#else
        "32"
#endif
#if defined DEBUG || defined _DEBUG
        ".Debug"
#endif
        ".state"
        ));
    MakeSlotAvailable(&fileNameSlot);
}


/*
 * job::PluginsStateFileGeneratorJob::~PluginsStateFileGeneratorJob
 */
job::PluginsStateFileGeneratorJob::~PluginsStateFileGeneratorJob() {
    Module::Release();
}


/*
 * job::PluginsStateFileGeneratorJob::create
 */
bool job::PluginsStateFileGeneratorJob::create(void) {
    return true; // intentionally empty ATM
}


/*
 * job::PluginsStateFileGeneratorJob::release
 */
void job::PluginsStateFileGeneratorJob::release(void) {
    // intentionally empty ATM
}

namespace {
    template<class F, class I, class T>
    void WriteSimpleTag(F& file, const I& indent, const T& tag, vislib::StringA& content) {
        if (!content.IsEmpty()) {
            vislib::xml::EncodeEntities(content);
            file << indent << "<" << tag << ">" << content << "</" << tag << ">" << std::endl;
        } else {
            file << indent << "<" << tag << " />" << std::endl;
        }
    }
    template<class F, class I, class T>
    void WriteSimpleTag(F& file, const I& indent, const T& tag, vislib::StringW& content) {
        vislib::StringA u;
        if (!content.IsEmpty()) vislib::UTF8Encoder::Encode(u, content);
        WriteSimpleTag(file, indent, tag, u);
    }
    template<class F, class I, class T>
    void WriteSimpleTag(F& file, const I& indent, const T& tag, const std::string& content) {
        vislib::StringA c(content.c_str());
        WriteSimpleTag(file, indent, tag, c);
    }
    template<class F, class I, class T>
    void WriteSimpleTag(F& file, const I& indent, const T& tag, const std::wstring& content) {
        vislib::StringA u;
        if (!content.empty()) vislib::UTF8Encoder::Encode(u, content.c_str());
        WriteSimpleTag(file, indent, tag, u);
    }
    template<class F, class I, class T1, class T2>
    void WriteEntityName(F& file, const I& indent, const T1& m, const T2& cs) {
        vislib::StringA str = cs->FullName();
        vislib::StringA str2 = m->FullName() + "::";
        if (str.StartsWith(str2)) str.Remove(0, str2.Length());
        vislib::xml::EncodeEntities(str);
        file << indent << "<Name>" << str << "</Name>" << std::endl;
    }
    template<class F, class I>
    void WriteParamCommonTypeInfoe(F& file, const I& indent, const param::AbstractParam * param) {
        vislib::RawStorage blob;
        param->Definition(blob);
        assert(blob.GetSize() >= 6);
        vislib::StringA tn(blob.As<const char>(), 6);
        WriteSimpleTag(file, indent, "TypeName", tn);
        vislib::TString defVal = param->ValueString();
        WriteSimpleTag(file, indent, "DefaultValue", defVal);
    }
}

/*
 * job::PluginsStateFileGeneratorJob::Run
 */
DWORD job::PluginsStateFileGeneratorJob::Run(void *userData) {
    auto& log = vislib::sys::Log::DefaultLog;
    log.WriteInfo("Collecting information for Plugins State File for MegaMol Configurator.");
    vislib::TString path = fileNameSlot.Param<param::FilePathParam>()->Value();
    path = vislib::sys::Path::Resolve(path);
    log.WriteInfo(vislib::TString(_T("Writing Plugins State File: ")) + path);

    std::ofstream file;
    file.open(path.PeekBuffer(), std::ios_base::out | std::ios_base::trunc);
    if (file.bad() || !file.is_open()) {
        log.WriteError("Failed to open file.");
        this->signalEnd(false);
        return -1;
    }

    file << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;

    file << "<MegaMolConfiguratorState xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">" << std::endl
        << "  <Version Version=\"1.0\" />" << std::endl
        << "  <Plugins>" << std::endl;

    CoreInstance *ci = GetCoreInstance();
    std::string name;
    const factories::ModuleDescriptionManager *modMan = nullptr;
    const factories::CallDescriptionManager *callMan = nullptr;

    // core:
    ci->GetAssemblyFileName(name);
    modMan = &ci->AbstractAssemblyInstance::GetModuleDescriptionManager();
    callMan = &ci->AbstractAssemblyInstance::GetCallDescriptionManager();
    log.WriteInfo("Info of \"%s\": %u modules, %u calls", name.c_str(), (modMan != nullptr) ? modMan->Count() : 0, (callMan != nullptr) ? callMan->Count() : 0);
    file << "    <PluginFile>" << std::endl
        << "      <IsCore>true</IsCore>" << std::endl;
    WriteSimpleTag(file, "      ", "Filename", name);
    writePluginInfo(file, modMan, callMan);
    file << "    </PluginFile>" << std::endl;

    // plugins
    const std::vector<utility::plugins::AbstractPluginInstance::ptr_type>& plugins = ci->Plugins().GetPlugins();
    for (utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
        plugin->GetAssemblyFileName(name);
        modMan = &plugin->GetModuleDescriptionManager();
        callMan = &plugin->GetCallDescriptionManager();
        log.WriteInfo("Info of \"%s\": %u modules, %u calls", name.c_str(), (modMan != nullptr) ? modMan->Count() : 0, (callMan != nullptr) ? callMan->Count() : 0);
        file << "    <PluginFile>" << std::endl
            << "      <IsCore>false</IsCore>" << std::endl;
        WriteSimpleTag(file, "      ", "Filename", name);
        writePluginInfo(file, modMan, callMan);
        file << "    </PluginFile>" << std::endl;
    }

    file << "  </Plugins>" << std::endl
        << "</MegaMolConfiguratorState>";

    file.close();
    log.WriteInfo("Finished writing state file.");

    this->signalEnd(this->shouldTerminate());
    return 0;
}

/*
 * job::PluginsStateFileGeneratorJob::writePluginInfo
 */
void job::PluginsStateFileGeneratorJob::writePluginInfo(std::ofstream& file,
        const factories::ModuleDescriptionManager *modMan,
        const factories::CallDescriptionManager *callMan) const {
    if ((modMan == nullptr) || (modMan->Count() <= 0)) {
        file << "      <Modules />" << std::endl;
    } else {
        file << "      <Modules>" << std::endl;
        auto mod_end = modMan->end();
        for (auto mod_i = modMan->begin(); mod_i != mod_end; ++mod_i) {
            file << "        <Module>" << std::endl;
            WriteModuleInfo(file, mod_i->get());
            file << "        </Module>" << std::endl;
        }
        file << "      </Modules>" << std::endl;
    }
    if ((callMan == nullptr) || (callMan->Count() <= 0)) {
        file << "      <Calls />" << std::endl;
    } else {
        file << "      <Calls>" << std::endl;
        auto call_end = callMan->end();
        for (auto call_i = callMan->begin(); call_i != call_end; ++call_i) {
            file << "        <Call>" << std::endl;
            WriteCallInfo(file, call_i->get());
            file << "        </Call>" << std::endl;
        }
        file << "      </Calls>" << std::endl;
    }
}

/*
 * job::PluginsStateFileGeneratorJob::WriteModuleInfo
 */
void job::PluginsStateFileGeneratorJob::WriteModuleInfo(std::ofstream& file,
        const factories::ModuleDescription* desc) const {
    CoreInstance *ci = GetCoreInstance();

    vislib::StringA Name = desc->ClassName();
    WriteSimpleTag(file, "          ", "Name", Name);

    vislib::StringA Description = desc->Description();
    Description.TrimSpaces();
    WriteSimpleTag(file, "          ", "Description", Description);

    megamol::core::RootModuleNamespace::ptr_type rms = std::make_shared<megamol::core::RootModuleNamespace>();
    megamol::core::Module::ptr_type m(desc->CreateModule(nullptr));
    if (m == nullptr) {
        file << "          <ParamSlots />" << std::endl
            << "          <CallerSlots />" << std::endl
            << "          <CalleeSlots />" << std::endl;
    } else{
        rms->AddChild(m);
        std::vector<std::shared_ptr<param::ParamSlot> > paramSlots;
        std::vector<std::shared_ptr<CallerSlot> > callerSlots;
        std::vector<std::shared_ptr<CalleeSlot> > calleeSlots;

        Module::child_list_type::iterator ano_end = m->ChildList_End();
        for (Module::child_list_type::iterator ano_i = m->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
            std::shared_ptr<param::ParamSlot> p = std::dynamic_pointer_cast<param::ParamSlot>(*ano_i);
            if (p) paramSlots.push_back(p);
            std::shared_ptr<CallerSlot> cr = std::dynamic_pointer_cast<CallerSlot>(*ano_i);
            if (cr) callerSlots.push_back(cr);
            std::shared_ptr<CalleeSlot> ce = std::dynamic_pointer_cast<CalleeSlot>(*ano_i);
            if (ce) calleeSlots.push_back(ce);
        }

        // ParamSlots
        if (paramSlots.size() > 0) {
            file << "          <ParamSlots>" << std::endl;
            for (std::shared_ptr<param::ParamSlot> p : paramSlots) {
                file << "            <ParamSlot>" << std::endl;

                WriteEntityName(file, "              ", m, p);
                vislib::StringA str = p->Description();
                str.TrimSpaces();
                WriteSimpleTag(file, "              ", "Description", str);

                WriteParamInfo(file, p->Parameter().operator->());

                file << "            </ParamSlot>" << std::endl;
            }
            file << "          </ParamSlots>" << std::endl;
        } else {
            file << "          <ParamSlots />" << std::endl;
        }

        // CallerSlots
        if (callerSlots.size() > 0) {
            file << "          <CallerSlots>" << std::endl;
            for (std::shared_ptr<CallerSlot> cs : callerSlots) {
                file << "            <CallerSlot>" << std::endl;

                WriteEntityName(file, "              ", m, cs);
                vislib::StringA str = cs->Description();
                str.TrimSpaces();
                WriteSimpleTag(file, "              ", "Description", str);

                SIZE_T callCount = cs->GetCompCallCount();
                if (callCount > 0) {
                    file << "              <CompatibleCalls>" << std::endl;
                    for (SIZE_T i = 0; i < callCount; ++i) {
                        file << "                <string>" 
                            << cs->GetCompCallClassName(i)
                            << "</string>" << std::endl;
                    }
                    file << "              </CompatibleCalls>" << std::endl;
                } else {
                    file << "              <CompatibleCalls />" << std::endl;
                }

                file << "            </CallerSlot>" << std::endl;
            }
            file << "          </CallerSlots>" << std::endl;
        } else {
            file << "          <CallerSlots />" << std::endl;
        }

        // CalleeSlots
        if (calleeSlots.size() > 0) {
            file << "          <CalleeSlots>" << std::endl;
            for (std::shared_ptr<CalleeSlot> cs : calleeSlots) {
                file << "            <CalleeSlot>" << std::endl;

                WriteEntityName(file, "              ", m, cs);
                vislib::StringA str = cs->Description();
                str.TrimSpaces();
                WriteSimpleTag(file, "              ", "Description", str);

                SIZE_T callbackCount = cs->GetCallbackCount();
                std::vector<std::string> callNames, funcNames;
                std::set<std::string> uniqueCallNames, completeCallNames;
                for (SIZE_T i = 0; i < callbackCount; ++i) {
                    uniqueCallNames.insert(cs->GetCallbackCallName(i));
                    callNames.push_back(cs->GetCallbackCallName(i));
                    funcNames.push_back(cs->GetCallbackFuncName(i));
                }
                size_t ll = callNames.size();
                assert(ll == funcNames.size());
                for (std::string callName : uniqueCallNames) {
                    factories::CallDescriptionManager::description_ptr_type desc = ci->GetCallDescriptionManager().Find(callName.c_str());
                    bool allFound = true;
                    if (desc != nullptr) {
                        for (unsigned int i = 0; i < desc->FunctionCount(); ++i) {
                            std::string fn = desc->FunctionName(i);
                            bool found = false;
                            for (size_t j = 0; j < ll; ++j) {
                                if ((callNames[j] == callName) && (funcNames[j] == fn)) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                allFound = false;
                                break;
                            }
                        }
                    } else {
                        allFound = false;
                    }
                    if (allFound) {
                        completeCallNames.insert(callName);
                    }
                }

                if (!completeCallNames.empty()) {
                    file << "              <CompatibleCalls>" << std::endl;
                    for (std::string callName : completeCallNames) {
                        file << "                <string>" << callName << "</string>" << std::endl;
                    }
                    file << "              </CompatibleCalls>" << std::endl;
                } else {
                    file << "              <CompatibleCalls />" << std::endl;
                }

                file << "            </CalleeSlot>" << std::endl;
            }
            file << "          </CalleeSlots>" << std::endl;
        } else {
            file << "          <CalleeSlots />" << std::endl;
        }

        paramSlots.clear();
        callerSlots.clear();
        calleeSlots.clear();
        rms->RemoveChild(m);
        m->SetAllCleanupMarks();
        m->PerformCleanup();
        m.reset();
    }
}

/*
 * job::PluginsStateFileGeneratorJob::WriteCallInfo
 */
void job::PluginsStateFileGeneratorJob::WriteCallInfo(std::ofstream& file,
        const factories::CallDescription* desc) const {
    vislib::StringA Name = desc->ClassName();
    vislib::xml::EncodeEntities(Name);
    file << "          <Name>" << Name << "</Name>" << std::endl;
    vislib::StringA Description = desc->Description();
    WriteSimpleTag(file, "          ", "Description", Description);
    if (desc->FunctionCount() > 0) {
        file << "          <FunctionName>" << std::endl;
        for (unsigned int i = 0; i < desc->FunctionCount(); ++i) {
            Name = desc->FunctionName(i);
            vislib::xml::EncodeEntities(Name);
            file << "            <string>" << Name << "</string>" << std::endl;
        }
        file << "          </FunctionName>" << std::endl;
    } else {
        file << "          <FunctionName />" << std::endl;
    }
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::AbstractParam* param) const {
    const param::BoolParam                   * p1 = dynamic_cast<const param::BoolParam                   *>(param);
    const param::ButtonParam                 * p2 = dynamic_cast<const param::ButtonParam                 *>(param);
    const param::EnumParam                   * p3 = dynamic_cast<const param::EnumParam                   *>(param);
    const param::FloatParam                  * p4 = dynamic_cast<const param::FloatParam                  *>(param);
    const param::IntParam                    * p5 = dynamic_cast<const param::IntParam                    *>(param);
    const param::FilePathParam               * p6 = dynamic_cast<const param::FilePathParam               *>(param);
    const param::FlexEnumParam               * p7 = dynamic_cast<const param::FlexEnumParam               *>(param);
    const param::ColorParam                  * p8 = dynamic_cast<const param::ColorParam                  *>(param);
    const param::LinearTransferFunctionParam * p9 = dynamic_cast<const param::LinearTransferFunctionParam *>(param);
    if (p1 != nullptr) { WriteParamInfo(file, p1); return; }
    if (p2 != nullptr) { WriteParamInfo(file, p2); return; }
    if (p3 != nullptr) { WriteParamInfo(file, p3); return; }
    if (p4 != nullptr) { WriteParamInfo(file, p4); return; }
    if (p5 != nullptr) { WriteParamInfo(file, p5); return; }
    if (p6 != nullptr) { WriteParamInfo(file, p6); return; }
    if (p7 != nullptr) { WriteParamInfo(file, p7); return; }
    if (p8 != nullptr) { WriteParamInfo(file, p8); return; }
    // fallback string:
    file << "              <Type xsi:type=\"String\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::BoolParam    * param) const {
    file << "              <Type xsi:type=\"Bool\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::ButtonParam  * param) const {
    file << "              <Type xsi:type=\"Button\">" << std::endl
        << "                <TypeName>MMBUTN</TypeName>" << std::endl
        << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::EnumParam    * param) const {
    file << "              <Type xsi:type=\"Enum\">" << std::endl
        << "                <TypeName>MMENUM</TypeName>" << std::endl
        << "                <DefaultValue>" << param->Value() << "</DefaultValue>" << std::endl
        << "                <Values>" << std::endl;
    vislib::Map<int, vislib::TString> valueMap = const_cast<param::EnumParam*>(param)->getMap();
    vislib::Map<int, vislib::TString>::Iterator valueMapIt = valueMap.GetIterator();
    while (valueMapIt.HasNext()) {
        file << "                  <int>" << valueMapIt.Next().Key() << "</int>" << std::endl;
    }
    file << "                </Values>" << std::endl
        << "                <ValueNames>" << std::endl;
    valueMapIt = valueMap.GetIterator();
    while (valueMapIt.HasNext()) {
        WriteSimpleTag(file, "                  ", "string", valueMapIt.Next().Value());
    }
    file << "                </ValueNames>" << std::endl
        << "              </Type>" << std::endl;
}

/*
* job::PluginsStateFileGeneratorJob::WriteParamInfo
*/
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::FlexEnumParam* param) const {
    file << "              <Type xsi:type=\"FlexEnum\">" << std::endl
        << "                <TypeName>MMFENU</TypeName>" << std::endl
        << "                <DefaultValue>" << param->Value().c_str() << "</DefaultValue>" << std::endl
        << "                <Values>" << std::endl;
    param::FlexEnumParam::Storage_t vals = const_cast<param::FlexEnumParam*>(param)->getStorage();
    for (auto &v: vals) {
        WriteSimpleTag(file, "                  ", "string", v);
    }
    file << "                </Values>" << std::endl
        << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::FloatParam   * param) const {
    file << "              <Type xsi:type=\"Float\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "                <MinValue>" << param->MinValue() << "</MinValue>" << std::endl
        << "                <MaxValue>" << param->MaxValue() << "</MaxValue>" << std::endl
        << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::IntParam     * param) const {
    file << "              <Type xsi:type=\"Int\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "                <MinValue>" << param->MinValue() << "</MinValue>" << std::endl
        << "                <MaxValue>" << param->MaxValue() << "</MaxValue>" << std::endl
        << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file, const param::FilePathParam* param) const {
    file << "              <Type xsi:type=\"FilePath\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void megamol::core::job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file,
    param::ColorParam const* param) const {
    file << "              <Type xsi:type=\"Color\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "              </Type>" << std::endl;
}

/*
 * job::PluginsStateFileGeneratorJob::WriteParamInfo
 */
void megamol::core::job::PluginsStateFileGeneratorJob::WriteParamInfo(std::ofstream& file,
    param::LinearTransferFunctionParam const* param) const {
    file << "              <Type xsi:type=\"TransferFunction\">" << std::endl;
    WriteParamCommonTypeInfoe(file, "                ", param);
    file << "              </Type>" << std::endl;
}
