/*
 * CoreInstance.cpp
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#if (_MSC_VER > 1000)
#pragma warning(disable : 4996)
#endif /* (_MSC_VER > 1000) */
#if (_MSC_VER > 1000)
#pragma warning(default : 4996)
#endif /* (_MSC_VER > 1000) */

#include <memory>
#include <string>

#include "mmcore/AbstractSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
//#include "mmcore/cluster/ClusterController.h"
#include "mmcore/factories/PluginRegister.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/profiler/Manager.h"
#include "mmcore/utility/buildinfo/BuildInfo.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/functioncast.h"
#include "vislib/net/AbstractSimpleMessage.h"
#include "vislib/net/NetworkInformation.h"
#include "vislib/net/Socket.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/PerformanceCounter.h"

#include "mmcore/utility/log/Log.h"
#include "png.h"
#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/MultiSz.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/Stack.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/functioncast.h"
#include "vislib/memutils.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/sysfunctions.h"

#include <sstream>

#include "mmcore/utility/graphics/ScreenShotComments.h"

/*****************************************************************************/

/*
 * megamol::core::CoreInstance::CoreInstance
 */
megamol::core::CoreInstance::CoreInstance(void)
        : config()
        , lua(nullptr)
        , namespaceRoot()
        , loadedLuaProjects()
        , paramUpdateListeners()
        , plugins()
        , all_call_descriptions()
        , all_module_descriptions() {

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Startup();
#endif /* ULTRA_SOCKET_STARTUP */

    this->namespaceRoot = std::make_shared<RootModuleNamespace>();

    profiler::Manager::Instance().SetCoreInstance(this);
    this->namespaceRoot->SetCoreInstance(*this);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance created");
}


/*
 * megamol::core::CoreInstance::~CoreInstance
 */
megamol::core::CoreInstance::~CoreInstance(void) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance destroyed");

    // Shutdown all views and jobs, which might still run
    {
        vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());
        AbstractNamedObjectContainer::child_list_type::iterator iter, end;
        while (true) {
            iter = this->namespaceRoot->ChildList_Begin();
            end = this->namespaceRoot->ChildList_End();
            if (iter == end)
                break;
            ModuleNamespace::ptr_type mn = ModuleNamespace::dynamic_pointer_cast(*iter);
            //        ModuleNamespace *mn = dynamic_cast<ModuleNamespace*>((*iter).get());
            this->closeViewJob(mn);
        }
    }

    // we need to manually clean up all data structures in the right order!
    // first view- and job-descriptions
    // then factories
    this->all_module_descriptions.Shutdown();
    this->all_call_descriptions.Shutdown();
    // finally plugins
    this->plugins.clear();

    delete this->lua;
    this->lua = nullptr;

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Cleanup();
#endif /* ULTRA_SOCKET_STARTUP */
}


/*
 * megamol::core::CoreInstance::GetCallDescriptionManager
 */
const megamol::core::factories::CallDescriptionManager& megamol::core::CoreInstance::GetCallDescriptionManager(
    void) const {
    return this->all_call_descriptions;
}


/*
 * megamol::core::CoreInstance::GetModuleDescriptionManager
 */
const megamol::core::factories::ModuleDescriptionManager& megamol::core::CoreInstance::GetModuleDescriptionManager(
    void) const {
    return this->all_module_descriptions;
}


/*
 * megamol::core::CoreInstance::Initialise
 */
void megamol::core::CoreInstance::Initialise() {

    this->lua = new LuaState(this);
    if (this->lua == nullptr || !this->lua->StateOk()) {
        throw vislib::IllegalStateException("Cannot initalise Lua", __FILE__, __LINE__);
    }
    std::string result;
    const bool ok = lua->RunString("mmLog(LOGINFO, 'Lua loaded OK: Running on ', "
                                   "mmGetBitWidth(), ' bit ', mmGetOS(), ' in ', mmGetConfiguration(),"
                                   "' mode on ', mmGetMachineName(), '.')",
        result);
    if (ok) {
        // megamol::core::utility::log::Log::DefaultLog.WriteInfo("Lua execution is OK and returned '%s'", result.c_str());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Lua execution is NOT OK and returned '%s'", result.c_str());
    }
    // lua->RunString("mmLogInfo('Lua loaded Ok.')");

    // loading plugins
    for (const auto& plugin : factories::PluginRegister::getAll()) {
        this->loadPlugin(plugin);
    }

    // set up profiling manager
    if (this->config.IsConfigValueSet("profiling")) {
        vislib::StringA prof(this->config.ConfigValue("profiling"));
        if (prof.Equals("all", false)) {
            profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_ALL);
        } else if (prof.Equals("selected", false)) {
            profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_SELECTED);
        } else if (prof.Equals("none", false)) {
            profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
        } else {
            try {
                bool b = vislib::CharTraitsA::ParseBool(prof);
                if (b) {
                    profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_SELECTED);
                } else {
                    profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
                }
            } catch (...) { profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE); }
        }
    } else {
        // Do not profile on default
        profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
    }

    //////////////////////////////////////////////////////////////////////

    translateShaderPaths(config);
}


/*
 * megamol::core::CoreInstance::FindParameter
 */
std::shared_ptr<megamol::core::param::AbstractParam> megamol::core::CoreInstance::FindParameter(
    const vislib::StringA& name, bool quiet, bool create) {
    using megamol::core::utility::log::Log;
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    vislib::Array<vislib::StringA> path = vislib::StringTokeniserA::Split(name, "::", true);
    vislib::StringA slotName("");
    if (path.Count() > 0) {
        slotName = path.Last();
        path.RemoveLast();
    }
    vislib::StringA modName("");
    if (path.Count() > 0) {
        modName = path.Last();
        path.RemoveLast();
    }

    ModuleNamespace::ptr_type mn;
    // parameter slots may have namespace operators in their names!
    while (!mn) {
        mn = this->namespaceRoot->FindNamespace(path, false, true);
        if (!mn) {
            if (path.Count() > 0) {
                slotName = modName + "::" + slotName;
                modName = path.Last();
                path.RemoveLast();
            } else {
                /*  if(create)
                    {
                        param::ParamSlot *slotNew = new param::ParamSlot(name, "newly inserted");
                        *slotNew << new param::StringParam("");
                        slotNew->MakeAvailable();
                        this->namespaceRoot.AddChild(slotNew);
                    }
                    else*/
                {
                    if (!quiet)
                        Log::DefaultLog.WriteError(
                            "Cannot find parameter \"%s\": namespace not found", name.PeekBuffer());
                    return NULL;
                }
            }
        }
    }

    Module::ptr_type mod = Module::dynamic_pointer_cast(mn->FindChild(modName));
    if (!mod) {
        /*  if(create)
            {
                param::ParamSlot *slot = new param::ParamSlot(name, "newly inserted");
                *slot << new param::StringParam("");
                slot->MakeAvailable();
                this->namespaceRoot.AddChild(slot);
                //mod = dynamic_cast<Module *>(this->namespaceRoot.FindChild(modName));
                return FindParameter(name, quiet, false);
            }
            else*/
        {
            if (!quiet)
                Log::DefaultLog.WriteError("Cannot find parameter \"%s\": module not found", name.PeekBuffer());
            return NULL;
        }
    }

    param::ParamSlot* slot = dynamic_cast<param::ParamSlot*>(mod->FindChild(slotName).get());
    if (slot == NULL) {
        /*  if(create)
            {
                param::ParamSlot *slotNew = new param::ParamSlot(name, "newly inserted");
                *slotNew << new param::StringParam("");
                slotNew->MakeAvailable();
                this->namespaceRoot.AddChild(slotNew);
                slot = slotNew;
            }
            else*/
        {
            if (!quiet)
                Log::DefaultLog.WriteError("Cannot find parameter \"%s\": slot not found", name.PeekBuffer());
            return NULL;
        }
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) {
        /*    if(create)
            {
                param::ParamSlot *slotNew = new param::ParamSlot(slotName, "newly inserted");
                *slotNew << new param::StringParam("");
                slotNew->MakeAvailable();
                this->namespaceRoot.AddChild(slotNew);
                slot = slotNew;
            }
            else*/
        {
            if (!quiet)
                Log::DefaultLog.WriteError("Cannot find parameter \"%s\": slot is not available", name.PeekBuffer());
            return NULL;
        }
    }
    if (slot->Parameter() == nullptr) {
        /*    if(create)
            {
                param::ParamSlot *slotNew = new param::ParamSlot(slotName, "newly inserted");
                *slotNew << new param::StringParam("");
                slotNew->MakeAvailable();
                this->namespaceRoot.AddChild(slotNew);
                slot = slotNew;
            }
            else*/
        {
            if (!quiet)
                Log::DefaultLog.WriteError("Cannot find parameter \"%s\": slot has no parameter", name.PeekBuffer());
            return NULL;
        }
    }


    return slot->Parameter();
}


std::string megamol::core::CoreInstance::SerializeGraph() {

    std::string serVersion =
        std::string("mmCheckVersion(\"") + megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() + "\")";
    std::string serInstances;
    std::string serModules;
    std::string serCalls;
    std::string serParams;

    std::stringstream confInstances, confModules, confCalls, confParams;

    std::map<std::string, std::string> view_instances;
    std::map<std::string, std::string> job_instances;
    {
        vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());
        AbstractNamedObjectContainer::ptr_type anoc =
            AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        int job_counter = 0;
        for (auto ano = anoc->ChildList_Begin(); ano != anoc->ChildList_End(); ++ano) {
            auto vi = dynamic_cast<ViewInstance*>(ano->get());
            auto ji = dynamic_cast<JobInstance*>(ano->get());
            if (vi && vi->View()) {
                std::string vin = vi->Name().PeekBuffer();
                view_instances[vi->View()->FullName().PeekBuffer()] = vin;
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "SerializeGraph: Found view instance \"%s\" with view \"%s\".",
                    view_instances[vi->View()->FullName().PeekBuffer()].c_str(), vi->View()->FullName().PeekBuffer());
            }
            if (ji && ji->Job()) {
                std::string jin = ji->Name().PeekBuffer();
                // todo: find job module! WTF!
                job_instances[jin] = std::string("job") + std::to_string(job_counter);
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "SerializeGraph: Found job instance \"%s\" with job \"%s\".", jin.c_str(),
                    job_instances[jin].c_str());
                ++job_counter;
            }
        }

        const auto fun = [&confInstances, &confModules, &confCalls, &confParams, &view_instances](Module* mod) {
            if (view_instances.find(mod->FullName().PeekBuffer()) != view_instances.end()) {
                confInstances << "mmCreateView(\"" << view_instances[mod->FullName().PeekBuffer()] << "\",\""
                              << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer() << "\")\n";
            } else {
                // todo: jobs??
                confModules << "mmCreateModule(\"" << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer()
                            << "\")\n";
            }
            AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
                 ++si) {
                const auto slot = dynamic_cast<param::ParamSlot*>((*si).get());
                if (slot) {
                    const auto bp = slot->Param<param::ButtonParam>();
                    if (!bp) {
                        std::string val = slot->Parameter()->ValueString();

                        // Encode to UTF-8 string
                        vislib::StringA valueString;
                        vislib::UTF8Encoder::Encode(valueString, vislib::StringA(val.c_str()));
                        val = valueString.PeekBuffer();

                        confParams << "mmSetParamValue(\"" << slot->FullName() << "\",[=[" << val << "]=])\n";
                    }
                }
                const auto cslot = dynamic_cast<CallerSlot*>((*si).get());
                if (cslot) {
                    const Call* c = const_cast<CallerSlot*>(cslot)->CallAs<Call>();
                    if (c != nullptr) {
                        confCalls << "mmCreateCall(\"" << c->ClassName() << "\",\""
                                  << c->PeekCallerSlot()->Parent()->FullName().PeekBuffer()
                                  << "::" << c->PeekCallerSlot()->Name().PeekBuffer() << "\",\""
                                  << c->PeekCalleeSlot()->Parent()->FullName().PeekBuffer()
                                  << "::" << c->PeekCalleeSlot()->Name().PeekBuffer() << "\")\n";
                    }
                }
            }
        };
        this->EnumModulesNoLock(nullptr, fun);

        serInstances = confInstances.str();
        serModules = confModules.str();
        serCalls = confCalls.str();
        serParams = confParams.str();
    }

    return serVersion + '\n' + serInstances + '\n' + serModules + '\n' + serCalls + '\n' + serParams;
}


void megamol::core::CoreInstance::EnumModulesNoLock(
    core::AbstractNamedObject* entry_point, std::function<void(Module*)> cb) {

    AbstractNamedObject* ano = entry_point;
    bool fromModule = true;
    if (!entry_point) {
        ano = this->namespaceRoot.get();
        fromModule = false;
    }

    auto anoc = dynamic_cast<AbstractNamedObjectContainer*>(ano);
    auto mod = dynamic_cast<Module*>(ano);
    std::vector<AbstractNamedObject*> anoStack;
    if (!fromModule || mod == nullptr) {
        // we start from the root or a namespace
        const auto it_end = anoc->ChildList_End();
        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
            if (dynamic_cast<AbstractNamedObjectContainer*>((*it).get())) {
                anoStack.push_back((*it).get());
            }
        }
        // if it was a namespace, we do not want to dig into calls!
        fromModule = false;
    } else {
        anoStack.push_back(anoc);
    }

    while (!anoStack.empty()) {
        ano = anoStack.back();
        anoStack.pop_back();

        anoc = dynamic_cast<AbstractNamedObjectContainer*>(ano);
        mod = dynamic_cast<Module*>(ano);

        if (mod) {
            cb(mod);
            if (fromModule) {
                const auto it_end = mod->ChildList_End();
                for (auto it = mod->ChildList_Begin(); it != it_end; ++it) {
                    auto cs = dynamic_cast<CallerSlot*>((*it).get());
                    if (cs) {
                        const Call* c = cs->CallAs<Call>();
                        if (c) {
                            this->FindModuleNoLock<core::Module>(c->PeekCalleeSlot()->Parent()->FullName().PeekBuffer(),
                                [&anoStack](core::Module* mod) { anoStack.push_back(mod); });
                        }
                    }
                }
            }
        } else if (anoc) {
            const auto it_end2 = anoc->ChildList_End();
            for (auto it = anoc->ChildList_Begin(); it != it_end2; ++it) {
                anoStack.push_back((*it).get());
            }
        }
    }
}

vislib::StringA megamol::core::CoreInstance::GetMergedLuaProject() const {
    if (this->loadedLuaProjects.Count() == 1) {
        return this->loadedLuaProjects[0].Second();
    }
    std::stringstream out;
    for (auto x = 0; x < this->loadedLuaProjects.Count(); x++) {
        out << this->loadedLuaProjects[x].Second();
        out << std::endl;
    }
    return out.str().c_str();
}


/*
 * megamol::core::CoreInstance::CleanupModuleGraph
 */
void megamol::core::CoreInstance::CleanupModuleGraph(void) {
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    this->namespaceRoot->SetAllCleanupMarks();

    AbstractNamedObjectContainer::child_list_type::iterator iter, end;
    end = this->namespaceRoot->ChildList_End();
    for (iter = this->namespaceRoot->ChildList_Begin(); iter != end; ++iter) {
        AbstractNamedObject::ptr_type child = *iter;
        ViewInstance* vi = dynamic_cast<ViewInstance*>(child.get());
        JobInstance* ji = dynamic_cast<JobInstance*>(child.get());

        if (vi != NULL) {
            vi->ClearCleanupMark();
        }
        if (ji != NULL) {
            ji->ClearCleanupMark();
        }
    }

    this->namespaceRoot->DisconnectCalls();
    this->namespaceRoot->PerformCleanup();
}


/*
 * megamol::core::CoreInstance::Shutdown
 */
void megamol::core::CoreInstance::Shutdown(void) {
    AbstractNamedObjectContainer::child_list_type::iterator iter, end;
    end = this->namespaceRoot->ChildList_End();
    for (iter = this->namespaceRoot->ChildList_Begin(); iter != end; ++iter) {
        AbstractNamedObject::ptr_type child = *iter;
        ViewInstance* vi = dynamic_cast<ViewInstance*>(child.get());
        JobInstance* ji = dynamic_cast<JobInstance*>(child.get());

        if (vi != NULL) {
            vi->RequestClose();
            vi->Terminate();
        }
        if (ji != NULL) {
            ji->Terminate();
        }
    }
}

/*
 * megamol::core::CoreInstance::ParameterValueUpdate
 */
void megamol::core::CoreInstance::ParameterValueUpdate(megamol::core::param::ParamSlot& slot) {
    vislib::SingleLinkedList<param::ParamUpdateListener*>::Iterator i = this->paramUpdateListeners.GetIterator();
    while (i.HasNext()) {
        i.Next()->ParamUpdated(slot);
    }
    this->paramUpdates.emplace_back(slot.FullName(), slot.Param<param::AbstractParam>()->ValueString());
}


/*
 * megamol::core::CoreInstance::closeViewJob
 */
void megamol::core::CoreInstance::closeViewJob(megamol::core::ModuleNamespace::ptr_type obj) {

    ASSERT(obj != NULL);
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    if (obj->Parent() != this->namespaceRoot) {
        // this happens when a job/view is removed from the graph before it's
        // handle is deletes (core instance destructor)
        return;
    }

    ViewInstance* vi = dynamic_cast<ViewInstance*>(obj.get());
    JobInstance* ji = dynamic_cast<JobInstance*>(obj.get());
    if (vi != NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "View instance %s terminating ...", vi->Name().PeekBuffer());
        vi->Terminate();
    }
    if (ji != NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "Job instance %s terminating ...", ji->Name().PeekBuffer());
        ji->Terminate();
    }

    ModuleNamespace::ptr_type mn = std::make_shared<ModuleNamespace>(obj->Name());

    while (obj->ChildList_Begin() != obj->ChildList_End()) {
        AbstractNamedObject::ptr_type ano = *obj->ChildList_Begin();
        obj->RemoveChild(ano);
        mn->AddChild(ano);
    }

    AbstractNamedObjectContainer::ptr_type p = AbstractNamedObjectContainer::dynamic_pointer_cast(obj->Parent());
    p->RemoveChild(obj);
    p->AddChild(mn);

    ASSERT(obj->Parent() == NULL);
    ASSERT(obj->ChildList_Begin() == obj->ChildList_End());

    this->CleanupModuleGraph();
}


/*
 * megamol::core::CoreInstance::loadPlugin
 */
void megamol::core::CoreInstance::loadPlugin(
    const std::shared_ptr<factories::AbstractPluginDescriptor>& pluginDescriptor) {

    try {

        auto new_plugin = pluginDescriptor->create();
        this->plugins.push_back(new_plugin);

        // report success
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Plugin \"%s\" loaded: %u Modules, %u Calls",
            new_plugin->GetObjectFactoryName().c_str(), new_plugin->GetModuleDescriptionManager().Count(),
            new_plugin->GetCallDescriptionManager().Count());

        for (auto md : new_plugin->GetModuleDescriptionManager()) {
            try {
                this->all_module_descriptions.Register(md);
            } catch (const std::invalid_argument&) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to load module description \"%s\": Naming conflict", md->ClassName());
            }
        }
        for (auto cd : new_plugin->GetCallDescriptionManager()) {
            try {
                this->all_call_descriptions.Register(cd);
            } catch (const std::invalid_argument&) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to load call description \"%s\": Naming conflict", cd->ClassName());
            }
        }

    } catch (const vislib::Exception& vex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to load Plugin: %s (%s, &d)", vex.GetMsgA(), vex.GetFile(), vex.GetLine());
    } catch (const std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to load Plugin: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to load Plugin: unknown exception");
    }
}


void megamol::core::CoreInstance::translateShaderPaths(megamol::core::utility::Configuration const& config) {
    auto const v_paths = config.ShaderDirectories();

    shaderPaths.resize(v_paths.Count());

    for (size_t idx = 0; idx < v_paths.Count(); ++idx) {
        shaderPaths[idx] = std::filesystem::path(v_paths[idx].PeekBuffer());
    }
}


std::vector<std::filesystem::path> megamol::core::CoreInstance::GetShaderPaths() const {
    return shaderPaths;
}
