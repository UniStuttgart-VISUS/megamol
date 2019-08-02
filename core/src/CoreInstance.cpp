/*
 * CoreInstance.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#if (_MSC_VER > 1000)
#    pragma warning(disable : 4996)
#endif /* (_MSC_VER > 1000) */
#include "vislib/graphics/gl/IncludeAllGL.h"
#if (_MSC_VER > 1000)
#    pragma warning(default : 4996)
#endif /* (_MSC_VER > 1000) */

#include <string>

#include "job/PluginsStateFileGeneratorJob.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/cluster/ClusterController.h"
#include "mmcore/cluster/ClusterViewMaster.h"
#include "mmcore/cluster/simple/Server.h"
#include "mmcore/job/JobThread.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/productversion.h"
#include "mmcore/profiler/Manager.h"
#include "mmcore/utility/APIValueUtil.h"
#include "mmcore/utility/ProjectParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "mmcore/versioninfo.h"
#include "vislib/GUID.h"
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
#include "vislib/sys/Log.h"
#include "vislib/sys/PerformanceCounter.h"
#include "vislib/sys/RegistryKey.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/vislibversion.h"

#include "factories/CallClassRegistry.h"
#include "factories/ModuleClassRegistry.h"
#include "utility/ServiceManager.h"
#include "utility/plugins/PluginManager.h"

#include "png.h"
#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/MultiSz.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/functioncast.h"
#include "vislib/memutils.h"
#include "vislib/sys/Console.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/sysfunctions.h"

#include <sstream>

#include "mmcore/utility/LuaHostService.h"

/*****************************************************************************/


/*
 * megamol::core::CoreInstance::PreInit::PreInit
 */
megamol::core::CoreInstance::PreInit::PreInit()
    : cfgFileSet(false)
    , logFileSet(false)
    , logLevelSet(false)
    , logEchoLevelSet(false)
    , cfgFile()
    , logFile()
    , logLevel(0)
    , logEchoLevel(0)
    , cfgOverrides()
    , cfgOverridesSet(false) {
    // atm intentionally empty
}


/*****************************************************************************/

#ifdef _WIN32
extern HMODULE mmCoreModuleHandle;
#endif

/*
 * megamol::core::CoreInstance::ViewJobHandleDalloc
 */
void megamol::core::CoreInstance::ViewJobHandleDalloc(void* data, megamol::core::ApiHandle* obj) {
    CoreInstance* core = reinterpret_cast<CoreInstance*>(data);
    if (core != NULL) {
        ModuleNamespace* vj = dynamic_cast<ModuleNamespace*>(obj);
        if (vj != NULL) {
            core->closeViewJob(ModuleNamespace::dynamic_pointer_cast(vj->shared_from_this()));
        }
    }
}


/*
 * megamol::core::CoreInstance::CoreInstance
 */
megamol::core::CoreInstance::CoreInstance(void)
    : ApiHandle()
    , factories::AbstractAssemblyInstance()
    , preInit(new PreInit)
    , config()
    , shaderSourceFactory(config)
    , log()
    , lua(nullptr)
    , builtinViewDescs()
    , projViewDescs()
    , builtinJobDescs()
    , projJobDescs()
    , pendingViewInstRequests()
    , pendingJobInstRequests()
    , namespaceRoot()
    , pendingCallInstRequests()
    , pendingCallDelRequests()
    , pendingModuleInstRequests()
    , pendingModuleDelRequests()
    , pendingParamSetRequests()
    , graphUpdateLock()
    , loadedLuaProjects()
    , timeOffset(0.0)
    , paramUpdateListeners()
    , plugins(nullptr)
    , all_call_descriptions()
    , all_module_descriptions()
    , parameterHash(1) {
    // setup log as early as possible.
    this->log.SetLogFileName(static_cast<const char*>(NULL), false);
    this->log.SetLevel(vislib::sys::Log::LEVEL_ALL);
#ifdef _DEBUG
    this->log.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
#else
    this->log.SetEchoLevel(vislib::sys::Log::LEVEL_ERROR);
#endif
    this->log.SetEchoTarget(new vislib::sys::Log::StreamTarget(stdout, vislib::sys::Log::LEVEL_ALL));
    this->log.SetOfflineMessageBufferSize(25);
    // redirect default log to instance log of last instance
    //  not perfect, but better than nothing.
    vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
    vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
    vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
    vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::RedirectTarget(&this->log));

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Startup();
#endif /* ULTRA_SOCKET_STARTUP */
    this->plugins = new utility::plugins::PluginManager();
    this->services = new utility::ServiceManager(*this);

#ifdef _WIN32
    WCHAR dll_path[MAX_PATH] = {0};
    GetModuleFileNameW(mmCoreModuleHandle, dll_path, _countof(dll_path));
    this->SetAssemblyFileName(dll_path);
#else
    this->SetAssemblyFileName("Core <TODO: Fix implementation>");
#endif

    this->namespaceRoot = std::make_shared<RootModuleNamespace>();

    profiler::Manager::Instance().SetCoreInstance(this);
    this->namespaceRoot->SetCoreInstance(*this);
    this->config.instanceLog = &this->log;
    factories::register_module_classes(this->module_descriptions);
    for (auto md : this->module_descriptions) this->all_module_descriptions.Register(md);
    factories::register_call_classes(this->call_descriptions);
    for (auto cd : this->call_descriptions) this->all_call_descriptions.Register(cd);

    // megamol::core::utility::LuaHostService::ID =
    //    this->InstallService<megamol::core::utility::LuaHostService>();

    // Normalize timer with time offset to something less crappy shitty hateworthy
    this->timeOffset = -this->GetCoreInstanceTime();

    //#ifdef _DEBUG
    // Use a randomized time offset to debug the offset synchronization
    { // use a guid to initialize the pseudo-random generator
        vislib::GUID guid;
        guid.Create();
        ::srand(guid.HashCode());
    }
    this->timeOffset += 100.0 * static_cast<double>(::rand()) / static_cast<double>(RAND_MAX);
    //#endif

    this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Core Instance created");

    // vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO+42, "GraphUpdateLock address: %x\n",
    // std::addressof(this->graphUpdateLock));
}


/*
 * megamol::core::CoreInstance::~CoreInstance
 */
megamol::core::CoreInstance::~CoreInstance(void) {
    this->config.instanceLog = NULL;
    SAFE_DELETE(this->preInit);
    this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Core Instance destroyed");

    // Shutdown all views and jobs, which might still run
    {
        vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());
        AbstractNamedObjectContainer::child_list_type::iterator iter, end;
        while (true) {
            iter = this->namespaceRoot->ChildList_Begin();
            end = this->namespaceRoot->ChildList_End();
            if (iter == end) break;
            ModuleNamespace::ptr_type mn = ModuleNamespace::dynamic_pointer_cast(*iter);
            //        ModuleNamespace *mn = dynamic_cast<ModuleNamespace*>((*iter).get());
            this->closeViewJob(mn);
        }
    }

    delete this->services;
    this->services = nullptr;

    // we need to manually clean up all data structures in the right order!
    // first view- and job-descriptions
    this->builtinViewDescs.Shutdown();
    this->builtinJobDescs.Shutdown();
    this->projViewDescs.Shutdown();
    this->projJobDescs.Shutdown();
    // then factories
    this->all_module_descriptions.Shutdown();
    this->all_call_descriptions.Shutdown();
    this->module_descriptions.Shutdown();
    this->call_descriptions.Shutdown();
    // finally plugins
    delete this->plugins;
    this->plugins = nullptr;

    delete this->lua;
    this->lua = nullptr;

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Cleanup();
#endif /* ULTRA_SOCKET_STARTUP */
}


/*
 * megamol::core::CoreInstance::GetAssemblyName
 */
const std::string& megamol::core::CoreInstance::GetAssemblyName(void) const {
    static std::string noname("");
    return noname;
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
void megamol::core::CoreInstance::Initialise(void) {
    if (this->preInit == NULL) {
        throw vislib::IllegalStateException("Cannot initialise a core instance twice.", __FILE__, __LINE__);
    }

    // logging mechanism
    if (this->preInit->IsLogEchoLevelSet()) {
        this->log.SetEchoLevel(this->preInit->GetLogEchoLevel());
        this->config.logEchoLevelLocked = true;
        if (this->preInit->GetLogEchoLevel() != 0) {
            this->log.EchoOfflineMessages(true);
        }
    }
    if (this->preInit->IsLogLevelSet()) {
        this->log.SetLevel(this->preInit->GetLogLevel());
        this->config.logLevelLocked = true;
        if (this->preInit->GetLogLevel() == 0) {
            this->log.SetLogFileName(static_cast<char*>(NULL), false);
            this->config.logFilenameLocked = true;
        } else {
            if (this->preInit->IsLogFileSet()) {
                this->log.SetLogFileName(this->preInit->GetLogFile(), false);
                this->config.logFilenameLocked = true;
            } else {
                this->log.SetLogFileName(static_cast<char*>(NULL), false);
            }
        }
    }
    vislib::sys::Log::DefaultLog.EchoOfflineMessages(true);

    this->lua = new LuaState(this);
    if (this->lua == nullptr || !this->lua->StateOk()) {
        throw vislib::IllegalStateException("Cannot initalise Lua", __FILE__, __LINE__);
    }
    std::string result;
    int ok = lua->RunString("mmLog(LOGINFO, 'Lua loaded OK: Running on ', "
                            "mmGetBitWidth(), ' bit ', mmGetOS(), ' in ', mmGetConfiguration(),"
                            "' mode on ', mmGetMachineName(), '.')",
        result);
    if (ok) {
        // vislib::sys::Log::DefaultLog.WriteInfo("Lua execution is OK and returned '%s'", result.c_str());
    } else {
        vislib::sys::Log::DefaultLog.WriteError("Lua execution is NOT OK and returned '%s'", result.c_str());
    }
    // lua->RunString("mmLogInfo('Lua loaded Ok.')");

    // configuration file
    if (this->preInit->IsConfigFileSet()) {
        this->config.LoadConfig(this->preInit->GetConfigFile());
    } else {
        this->config.LoadConfig();
    }

    // config overrides from command line
    if (this->preInit->IsConfigOverrideSet()) {
        const vislib::StringW& overrides = this->preInit->GetConfigFileOverrides();
        int pos = 0;
        int next = overrides.Find('\b', pos);
        do {
            if (next == vislib::StringW::INVALID_POS) next = overrides.Length();
            auto sub = overrides.Substring(pos, next - pos);
            int split = sub.Find('\a');
            if (split != vislib::StringW::INVALID_POS) {
                auto name = sub.Substring(0, split);
                auto val = sub.Substring(split + 1, sub.Length() - split);
                vislib::sys::Log::DefaultLog.WriteWarn("Overriding from command line:");
                this->config.SetValue<wchar_t>(MMC_CFGID_VARIABLE, name, val);
            }
            pos = next + 1;
        } while ((next = overrides.Find('\b', pos)) != vislib::StringW::INVALID_POS || pos < overrides.Length());
    }

    // register services? TODO: right place?
    megamol::core::utility::LuaHostService::ID = this->InstallService<megamol::core::utility::LuaHostService>();

    // loading plugins
    // printf("Log: %d:\n", (long)(&vislib::sys::Log::DefaultLog));
    // printf("\tAutoflush: %s\n", vislib::sys::Log::DefaultLog.IsAutoFlushEnabled() ? "enabled" : "disabled");
    // printf("\tLevel: %u\n", vislib::sys::Log::DefaultLog.GetLevel());
    // printf("\tEcho-Level: %u\n", vislib::sys::Log::DefaultLog.GetEchoLevel());
    // printf("\tEcho-Target: %d\n", (long)(vislib::sys::Log::DefaultLog.GetEchoOutTarget()));
    vislib::SingleLinkedList<vislib::TString> plugins_paths;
    this->config.ListPluginsToLoad(plugins_paths);
    vislib::SingleLinkedList<vislib::TString>::Iterator iter = plugins_paths.GetIterator();
    while (iter.HasNext()) {
        this->loadPlugin(iter.Next());
    }
    // printf("Log: %d:\n", (long)(&vislib::sys::Log::DefaultLog));
    // printf("\tAutoflush: %s\n", vislib::sys::Log::DefaultLog.IsAutoFlushEnabled() ? "enabled" : "disabled");
    // printf("\tLevel: %u\n", vislib::sys::Log::DefaultLog.GetLevel());
    // printf("\tEcho-Level: %u\n", vislib::sys::Log::DefaultLog.GetEchoLevel());
    // printf("\tEcho-Target: %d\n", (long)(vislib::sys::Log::DefaultLog.GetEchoOutTarget()));

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
            } catch (...) {
                profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
            }
        }
    } else {
        // Do not profile on default
        profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
    }


    //////////////////////////////////////////////////////////////////////
    // register builtin descriptions
    //////////////////////////////////////////////////////////////////////
    // view descriptions
    //////////////////////////////////////////////////////////////////////
    std::shared_ptr<ViewDescription> vd;

    // empty view; name for compatibility reasons
    vd = std::make_shared<ViewDescription>("emptyview");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty View3D
    vd = std::make_shared<ViewDescription>("emptyview3d");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty View2D
    vd = std::make_shared<ViewDescription>("emptyview2d");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View2D"), "view");
    // 'View2D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty view (show the title); name for compatibility reasons
    vd = std::make_shared<ViewDescription>("titleview");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // view for powerwall
    vd = std::make_shared<ViewDescription>("powerwallview");
    vd->AddModule(this->GetModuleDescriptionManager().Find("PowerwallView"), "pwview");
    // vd->AddModule(this->GetModuleDescriptionManager().Find("ClusterController"), "::cctrl"); // TODO: Dependant
    // instance!
    vd->AddCall(
        this->GetCallDescriptionManager().Find("CallRegisterAtController"), "pwview::register", "::cctrl::register");
    vd->SetViewModuleID("pwview");
    this->builtinViewDescs.Register(vd);

    // view for fusionex-hack (client side)
    vd = std::make_shared<ViewDescription>("simpleclusterview");
    vd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterClient"), "::scc");
    vd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterView"), "scview");
    vd->AddCall(this->GetCallDescriptionManager().Find("SimpleClusterClientViewRegistration"), "scview::register",
        "::scc::registerView");
    vd->SetViewModuleID("scview");

    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "::logo");
    vd->AddCall(this->GetCallDescriptionManager().Find("CallRenderView"), "scview::renderView", "::logo::render");

    this->builtinViewDescs.Register(vd);

    vd = std::make_shared<ViewDescription>("mpiclusterview");
    vd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterClient"), "::mcc");
    vd->AddModule(this->GetModuleDescriptionManager().Find("MpiProvider"), "::mpi");
    vd->AddModule(this->GetModuleDescriptionManager().Find("MpiClusterView"), "mcview");
    vd->AddCall(this->GetCallDescriptionManager().Find("SimpleClusterClientViewRegistration"), "mcview::register",
        "::mcc::registerView");
    vd->AddCall(this->GetCallDescriptionManager().Find("MpiCall"), "mcview::requestMpi", "::mpi::provideMpi");
    vd->SetViewModuleID("mcview");

    this->builtinViewDescs.Register(vd);

    // test view for sphere rendering
    vd = std::make_shared<ViewDescription>("testspheres");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "view");
    vd->AddModule(this->GetModuleDescriptionManager().Find("SphereRenderer"), "rnd");
    vd->AddModule(this->GetModuleDescriptionManager().Find("TestSpheresDataSource"), "dat");
    vd->AddCall(this->GetCallDescriptionManager().Find("CallRender3D"), "view::rendering", "rnd::rendering");
    vd->AddCall(this->GetCallDescriptionManager().Find("MultiParticleDataCall"), "rnd::getData", "dat::getData");
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // test view for sphere rendering
    vd = std::make_shared<ViewDescription>("testgeospheres");
    vd->AddModule(this->GetModuleDescriptionManager().Find("View3D"), "view");
    vd->AddModule(this->GetModuleDescriptionManager().Find("SimpleGeoSphereRenderer"), "rnd");
    vd->AddModule(this->GetModuleDescriptionManager().Find("TestSpheresDataSource"), "dat");
    vd->AddCall(this->GetCallDescriptionManager().Find("CallRender3D"), "view::rendering", "rnd::rendering");
    vd->AddCall(this->GetCallDescriptionManager().Find("MultiParticleDataCall"), "rnd::getData", "dat::getData");
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    //////////////////////////////////////////////////////////////////////
    // job descriptions
    //////////////////////////////////////////////////////////////////////
    std::shared_ptr<JobDescription> jd;

    // job for the cluster controller modules
    jd = std::make_shared<JobDescription>("clustercontroller");
    jd->AddModule(this->GetModuleDescriptionManager().Find("ClusterController"), "::cctrl");
    jd->SetJobModuleID("::cctrl");
    this->builtinJobDescs.Register(jd);

    // job for the cluster controller head-node modules
    jd = std::make_shared<JobDescription>("clusterheadcontroller");
    jd->AddModule(this->GetModuleDescriptionManager().Find("ClusterController"), "::cctrl");
    jd->AddModule(this->GetModuleDescriptionManager().Find("ClusterViewMaster"), "::cmaster");
    jd->AddCall(
        this->GetCallDescriptionManager().Find("CallRegisterAtController"), "::cmaster::register", "::cctrl::register");
    jd->SetJobModuleID("::cctrl");
    this->builtinJobDescs.Register(jd);

    // job for the cluster display client heartbeat server
    jd = std::make_shared<JobDescription>("heartbeat");
    jd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterClient"), "::scc");
    jd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterHeartbeat"), "scheartbeat");
    jd->AddCall(this->GetCallDescriptionManager().Find("SimpleClusterClientViewRegistration"), "scheartbeat::register",
        "::scc::registerView");
    jd->SetJobModuleID("scheartbeatthread");
    this->builtinJobDescs.Register(jd);

    // view for fusionex-hack (server side)
    jd = std::make_shared<JobDescription>("simpleclusterserver");
    jd->AddModule(this->GetModuleDescriptionManager().Find("SimpleClusterServer"), "::scs");
    jd->SetJobModuleID("::scs");
    this->builtinJobDescs.Register(jd);

    // // TODO: Replace (is deprecated)
    // job to produce images
    jd = std::make_shared<JobDescription>("imagemaker");
    jd->AddModule(this->GetModuleDescriptionManager().Find("ScreenShooter"), "imgmaker");
    jd->SetJobModuleID("imgmaker");
    this->builtinJobDescs.Register(jd);

    // TODO: Debug
    jd = std::make_shared<JobDescription>("DEBUGjob");
    jd->AddModule(this->GetModuleDescriptionManager().Find("JobThread"), "ctrl");
    jd->SetJobModuleID("ctrl");
    this->builtinJobDescs.Register(jd);

    // Generate PluginsStateFile for MegaMol Configurator:
    jd = std::make_shared<JobDescription>("GenStateFile");
    jd->AddModule(this->GetModuleDescriptionManager().Find("PluginsStateFileGeneratorJob"), "gen");
    jd->SetJobModuleID("gen");
    this->builtinJobDescs.Register(jd);

    //////////////////////////////////////////////////////////////////////


    while (this->config.HasInstantiationRequests()) {
        utility::Configuration::InstanceRequest r = this->config.GetNextInstantiationRequest();

        std::shared_ptr<const megamol::core::ViewDescription> vd =
            this->FindViewDescription(vislib::StringA(r.Description()));
        if (vd) {
            this->RequestViewInstantiation(vd.get(), r.Identifier(), &r);
            continue;
        }
        std::shared_ptr<const megamol::core::JobDescription> jd =
            this->FindJobDescription(vislib::StringA(r.Description()));
        if (jd) {
            this->RequestJobInstantiation(jd.get(), r.Identifier(), &r);
            continue;
        }

        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Unable to instance \"%s\" as \"%s\": Description not found.\n",
            vislib::StringA(r.Description()).PeekBuffer(), vislib::StringA(r.Identifier()).PeekBuffer());
    }

    SAFE_DELETE(this->preInit);
}


/*
 * megamol::core::CoreInstance::SetInitValue
 */
mmcErrorCode megamol::core::CoreInstance::SetInitValue(mmcInitValue key, mmcValueType type, const void* value) {
    if (this->preInit == NULL) {
        throw vislib::IllegalStateException("Core instance already initialised.", __FILE__, __LINE__);
    }

    try {
        switch (key) {
        case MMC_INITVAL_CFGFILE:
            if (!utility::APIValueUtil::IsStringType(type)) {
                return MMC_ERR_TYPE;
            }
            this->preInit->SetConfigFile(utility::APIValueUtil::AsStringW(type, value));
            break;
        case MMC_INITVAL_LOGFILE:
            if (!utility::APIValueUtil::IsStringType(type)) {
                return MMC_ERR_TYPE;
            }
            this->preInit->SetLogFile(utility::APIValueUtil::AsStringW(type, value));
            break;
        case MMC_INITVAL_LOGLEVEL:
            if (!utility::APIValueUtil::IsIntType(type)) {
                return MMC_ERR_TYPE;
            }
            this->preInit->SetLogLevel(utility::APIValueUtil::AsUint32(type, value));
            break;
        case MMC_INITVAL_LOGECHOLEVEL:
            if (!utility::APIValueUtil::IsIntType(type)) {
                return MMC_ERR_TYPE;
            }
            this->preInit->SetLogEchoLevel(utility::APIValueUtil::AsUint32(type, value));
            break;
        case MMC_INITVAL_INCOMINGLOG: {
            // if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
            // vislib::sys::Log *log = static_cast<vislib::sys::Log*>(
            //    const_cast<void*>(value));
            // log->SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
            // log->SetEchoTarget(&this->logRedirection);
            // log->EchoOfflineMessages(true);
            // log->SetLogFileName(static_cast<const char*>(NULL), false);
            // log->SetLevel(vislib::sys::Log::LEVEL_NONE);
            return MMC_ERR_NOT_IMPLEMENTED; // use MMC_INITVAL_CORELOG instead
        } break;
        case MMC_INITVAL_CORELOG: {
            if (type != MMC_TYPE_VOIDP) {
                return MMC_ERR_TYPE;
            }
            vislib::sys::Log** log = static_cast<vislib::sys::Log**>(const_cast<void*>(value));
            *log = &this->log;
        } break;
        case MMC_INITVAL_LOGECHOFUNC:
            if (type != MMC_TYPE_VOIDP) {
                return MMC_ERR_TYPE;
            }
            this->log.SetEchoTarget(
                new utility::LogEchoTarget(function_cast<mmcLogEchoFunction>(const_cast<void*>(value))));
            break;
        case MMC_INITVAL_CFGOVERRIDE:
            if (!utility::APIValueUtil::IsStringType(type)) {
                return MMC_ERR_TYPE;
            }
            this->preInit->SetConfigFileOverrides(utility::APIValueUtil::AsStringW(type, value));
            break;
        default:
            return MMC_ERR_UNKNOWN;
        }
    } catch (...) {
        return MMC_ERR_UNKNOWN;
    }
    return MMC_ERR_NO_ERROR;
}


/*
 * megamol::core::CoreInstance::FindViewDescription
 */
std::shared_ptr<const megamol::core::ViewDescription> megamol::core::CoreInstance::FindViewDescription(
    const char* name) {
    std::shared_ptr<const ViewDescription> d = NULL;
    if (d == NULL) {
        d = this->projViewDescs.Find(name);
    }
    if (d == NULL) {
        d = this->builtinViewDescs.Find(name);
    }
    return d;
}


/*
 * megamol::core::CoreInstance::EnumViewDescriptions
 */
void megamol::core::CoreInstance::EnumViewDescriptions(mmcEnumStringAFunction func, void* data, bool getBuiltinToo) {
    assert(func);
    for (auto vd : this->projViewDescs) {
        func(vd->ClassName(), data);
    }
    if (getBuiltinToo) {
        for (auto vd : this->builtinViewDescs) {
            func(vd->ClassName(), data);
        }
    }
}


/*
 * megamol::core::CoreInstance::FindJobDescription
 */
std::shared_ptr<const megamol::core::JobDescription> megamol::core::CoreInstance::FindJobDescription(const char* name) {
    std::shared_ptr<const JobDescription> d;
    if (!d) d = this->projJobDescs.Find(name);
    if (!d) d = this->builtinJobDescs.Find(name);
    return d;
}


/*
 * megamol::core::CoreInstance::RequestAllInstantiations
 */
void megamol::core::CoreInstance::RequestAllInstantiations() {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    for (auto vd : this->projViewDescs) {
        int cnt = static_cast<int>(this->pendingViewInstRequests.Count());
        std::string s = std::to_string(cnt);
        vislib::StringA name = "v";
        name.Append(s.c_str());
        ViewInstanceRequest req;
        req.SetName(name);
        req.SetDescription(vd.get());
        this->pendingViewInstRequests.Add(req);
    }
    for (auto jd : this->projJobDescs) {
        int cnt = static_cast<int>(this->pendingJobInstRequests.Count());
        std::string s = std::to_string(cnt);
        vislib::StringA name = "j";
        name.Append(s.c_str());
        JobInstanceRequest req;
        req.SetName(name);
        req.SetDescription(jd.get());
        this->pendingJobInstRequests.Add(req);
    }
}


/*
 * megamol::core::CoreInstance::RequestViewInstantiation
 */
void megamol::core::CoreInstance::RequestViewInstantiation(
    const megamol::core::ViewDescription* desc, const vislib::StringA& id, const ParamValueSetRequest* param) {
    if (id.Find(':') != vislib::StringA::INVALID_POS) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "View instantiation request aborted: name contains invalid character \":\"");
        return;
    }
    // could check here if the description is instantiable, but I do not want
    // to.
    ASSERT(desc);
    ViewInstanceRequest req;
    req.SetName(id);
    req.SetDescription(desc);
    if (param != NULL) {
        static_cast<ParamValueSetRequest&>(req) = *param;
    }
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingViewInstRequests.Add(req);
}


/*
 * megamol::core::CoreInstance::RequestJobInstantiation
 */
void megamol::core::CoreInstance::RequestJobInstantiation(
    const megamol::core::JobDescription* desc, const vislib::StringA& id, const ParamValueSetRequest* param) {
    if (id.Find(':') != vislib::StringA::INVALID_POS) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Job instantiation request aborted: name contains invalid character \":\"");
        return;
    }
    // could check here if the description is instantiable, but I do not want
    // to.
    ASSERT(desc);
    JobInstanceRequest req;
    req.SetName(id);
    req.SetDescription(desc);
    if (param != NULL) {
        static_cast<ParamValueSetRequest&>(req) = *param;
    }
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingJobInstRequests.Add(req);
}


bool megamol::core::CoreInstance::RequestModuleDeletion(const vislib::StringA& id) {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingModuleDelRequests.Add(id);
    return true;
}


bool megamol::core::CoreInstance::RequestCallDeletion(const vislib::StringA& from, const vislib::StringA& to) {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingCallDelRequests.Add(vislib::Pair<vislib::StringA, vislib::StringA>(from, to));
    return true;
}


bool megamol::core::CoreInstance::RequestModuleInstantiation(
    const vislib::StringA& className, const vislib::StringA& id) {

    factories::ModuleDescription::ptr md = this->GetModuleDescriptionManager().Find(vislib::StringA(className));
    if (md == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to request instantiation of module"
                                                " \"%s\": class \"%s\" not found.",
            id.PeekBuffer(), className.PeekBuffer());
        return false;
    }

    core::InstanceDescription::ModuleInstanceRequest mir;
    mir.SetFirst(id);
    mir.SetSecond(md);
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingModuleInstRequests.Add(mir);
    return true;
}


bool megamol::core::CoreInstance::RequestCallInstantiation(
    const vislib::StringA& className, const vislib::StringA& from, const vislib::StringA& to) {

    factories::CallDescription::ptr cd = this->GetCallDescriptionManager().Find(vislib::StringA(className));
    if (cd == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to request instantiation of "
                                                " unknown call class \"%s\".",
            className.PeekBuffer());
        return false;
    }

    core::InstanceDescription::CallInstanceRequest cir(from, to, cd, false);
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingCallInstRequests.Add(cir);
    return true;
}

bool megamol::core::CoreInstance::RequestChainCallInstantiation(
    const vislib::StringA& className, const vislib::StringA& chainStart, const vislib::StringA& to) {
    factories::CallDescription::ptr cd = this->GetCallDescriptionManager().Find(vislib::StringA(className));
    if (cd == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to request chain instantiation of "
                                                " unknown call class \"%s\".",
            className.PeekBuffer());
        return false;
    }

    core::InstanceDescription::CallInstanceRequest cir(chainStart, to, cd, false);
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingChainCallInstRequests.Add(cir);
    return true;
}


bool megamol::core::CoreInstance::RequestParamValue(const vislib::StringA& id, const vislib::StringA& value) {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    this->pendingParamSetRequests.Add(vislib::Pair<vislib::StringA, vislib::StringA>(id, value));
    return true;
}

bool megamol::core::CoreInstance::CreateParamGroup(const vislib::StringA& name, const int size) {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    if (this->pendingGroupParamSetRequests.Contains(name)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Cannot create parameter group %s: group already exists!", name.PeekBuffer());
        return false;
    } else {
        ParamGroup pg;
        pg.GroupSize = size;
        pg.Name = name;
        this->pendingGroupParamSetRequests[name] = pg;
        vislib::sys::Log::DefaultLog.WriteInfo("Created parameter group %s with size %i", name.PeekBuffer(), size);
        return true;
    }
}

bool megamol::core::CoreInstance::RequestParamGroupValue(
    const vislib::StringA& group, const vislib::StringA& id, const vislib::StringA& value) {

    vislib::sys::AutoLock l(this->graphUpdateLock);
    if (this->pendingGroupParamSetRequests.Contains(group)) {
        auto& g = this->pendingGroupParamSetRequests[group];
        if (g.Requests.Contains(id)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Cannot queue %s parameter change in group %s twice!", id.PeekBuffer(), group.PeekBuffer());
            return false;
        } else {
            vislib::sys::Log::DefaultLog.WriteInfo("Queueing parameter value change: [%s] %s = %s", group.PeekBuffer(),
                id.PeekBuffer(), value.PeekBuffer());
            g.Requests[id] = value;
        }
        return true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Cannot queue parameter change into nonexisting group %s", group.PeekBuffer());
        return false;
    }
}


bool megamol::core::CoreInstance::FlushGraphUpdates() {
    vislib::sys::AutoLock u(this->graphUpdateLock);

    if (this->pendingCallInstRequests.Count() != 0)
        this->callInstRequestsFlushIndices.push_back(this->pendingCallInstRequests.Count() - 1);
    if (this->pendingChainCallInstRequests.Count() != 0)
        this->chainCallInstRequestsFlushIndices.push_back(this->pendingChainCallInstRequests.Count() - 1);
    if (this->pendingModuleInstRequests.Count() != 0)
        this->moduleInstRequestsFlushIndices.push_back(this->pendingModuleInstRequests.Count() - 1);
    if (this->pendingCallDelRequests.Count() != 0)
        this->callDelRequestsFlushIndices.push_back(this->pendingCallDelRequests.Count() - 1);
    if (this->pendingModuleDelRequests.Count() != 0)
        this->moduleDelRequestsFlushIndices.push_back(this->pendingModuleDelRequests.Count() - 1);
    if (this->pendingParamSetRequests.Count() != 0)
        this->paramSetRequestsFlushIndices.push_back(this->pendingParamSetRequests.Count() - 1);
    if (this->pendingGroupParamSetRequests.Count() != 0)
        this->groupParamSetRequestsFlushIndices.push_back(this->pendingGroupParamSetRequests.Count() - 1);

    return true;
}


void megamol::core::CoreInstance::PerformGraphUpdates() {
    vislib::sys::AutoLock u(this->graphUpdateLock);
    vislib::sys::AutoLock m(this->ModuleGraphRoot()->ModuleGraphLock());

    AbstractNamedObject::ptr_type ano = this->namespaceRoot;
    AbstractNamedObjectContainer::ptr_type root = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(ano);
    if (!root) {
        vislib::sys::Log::DefaultLog.WriteError("PerformGraphUpdates: no root");
        return;
    }

    // counts processed graph update events for flush mechanism
    size_t counter = 0;

    this->shortenFlushIdxList(this->pendingModuleDelRequests.Count(), this->moduleDelRequestsFlushIndices);

    // delete modules
    while (this->pendingModuleDelRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->moduleDelRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->moduleDelRequestsFlushIndices);
            break;
        }

        auto mdr = this->pendingModuleDelRequests.First();
        this->pendingModuleDelRequests.RemoveFirst();

        ++counter;

        Module::ptr_type mod = Module::dynamic_pointer_cast(root.get()->FindNamedObject(mdr));
        if (!mod) {
            vislib::sys::Log::DefaultLog.WriteError(
                "PerformGraphUpdates: could not find module \"%s\" for deletion.", mdr.PeekBuffer());
            continue;
        }

        if (mod.get()->Parent() != nullptr) {
            auto p = mod.get()->Parent();
            auto n = dynamic_cast<core::ModuleNamespace*>(p.get());
            if (n) {

                // find all incoming calls. these need to be disconnected properly and deleted.
                std::vector<AbstractNamedObjectContainer::ptr_type> anoStack;
                anoStack.push_back(root);
                while (anoStack.size() > 0) {
                    AbstractNamedObjectContainer::ptr_type anoc = anoStack.back();
                    anoStack.pop_back();

                    if (anoc) {
                        auto it_end = anoc->ChildList_End();
                        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
                            AbstractNamedObject::ptr_type ano = *it;
                            AbstractNamedObjectContainer::ptr_type anoc =
                                std::dynamic_pointer_cast<AbstractNamedObjectContainer>(ano);
                            if (anoc) {
                                anoStack.push_back(anoc);
                            } else {
                                core::CallerSlot* callerSlot = dynamic_cast<core::CallerSlot*>((*it).get());
                                if (callerSlot != nullptr) {
                                    core::Call* call = callerSlot->CallAs<Call>();
                                    if (call != nullptr) {
                                        auto target = call->PeekCalleeSlot()->Parent();
                                        if (target->FullName().Equals(mod->FullName())) {
                                            // this call points to mod
                                            // vislib::sys::Log::DefaultLog.WriteInfo("found call from %s to %s",
                                            // call->PeekCallerSlot()->FullName(),
                                            //   call->PeekCalleeSlot()->FullName());
                                            callerSlot->ConnectCall(nullptr);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // delete all outgoing calls of mod
                AbstractNamedObjectContainer::child_list_type::iterator children, childrenend;
                childrenend = mod->ChildList_End();
                std::vector<AbstractNamedObject::ptr_type> deletionQueue;
                for (children = mod->ChildList_Begin(); children != childrenend; ++children) {
                    AbstractNamedObject::ptr_type child = *children;
                    AbstractNamedObjectContainer::ptr_type anoc =
                        AbstractNamedObjectContainer::dynamic_pointer_cast(child);
                    if (anoc) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "PerformGraphUpdates: found container \"%s\" inside module \"%s\"",
                            anoc->FullName().PeekBuffer(), mod->FullName().PeekBuffer());
                        continue;
                    }
                    core::CallerSlot* callerSlot = dynamic_cast<core::CallerSlot*>(child.get());
                    if (callerSlot != nullptr) {
                        core::Call* call = callerSlot->CallAs<Call>();
                        if (call != nullptr) {
                            vislib::sys::Log::DefaultLog.WriteInfo("removing call from %s to %s",
                                call->PeekCallerSlot()->FullName().PeekBuffer(),
                                call->PeekCalleeSlot()->FullName().PeekBuffer());
                            callerSlot->ConnectCall(nullptr);
                        }
                        deletionQueue.push_back(child);
                    }
                    core::param::ParamSlot* paramSlot = dynamic_cast<core::param::ParamSlot*>(child.get());
                    if (paramSlot != nullptr) {
                        // paramSlot->SetCleanupMark(true);
                        // paramSlot->PerformCleanup();
                        deletionQueue.push_back(child);
                    }
                    core::CalleeSlot* calleeSlot = dynamic_cast<core::CalleeSlot*>(child.get());
                    if (calleeSlot != nullptr) {
                        deletionQueue.push_back(child);
                    }
                }

                for (auto& c : deletionQueue) {
                    mod->RemoveChild(c);
                }

                for (children = mod->ChildList_Begin(); children != childrenend; ++children) {
                    AbstractNamedObject::ptr_type child = *children;
                    vislib::sys::Log::DefaultLog.WriteError(
                        "child remaining in %s: %s", mod->FullName().PeekBuffer(), child->FullName().PeekBuffer());
                }

                // remove mod
                n->RemoveChild(mod);
            } else {
                vislib::sys::Log::DefaultLog.WriteError("PerformGraphUpdates:module \"%s\" has no parent of type "
                                                        "ModuleNamespace. Deletion makes no sense.",
                    mdr.PeekBuffer());
                continue;
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "PerformGraphUpdates:module \"%s\" has no parent. Deletion makes no sense.", mdr.PeekBuffer());
            continue;
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingCallDelRequests.Count(), this->callDelRequestsFlushIndices);

    // delete calls
    while (this->pendingCallDelRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->callDelRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->callDelRequestsFlushIndices);
            break;
        }

        auto cdr = this->pendingCallDelRequests.First();
        this->pendingCallDelRequests.RemoveFirst();

        ++counter;

        bool found = false;
        // find the call
        std::vector<AbstractNamedObjectContainer::ptr_type> anoStack;
        anoStack.push_back(root);
        while (anoStack.size() > 0) {
            AbstractNamedObjectContainer::ptr_type anoc = anoStack.back();
            anoStack.pop_back();

            if (anoc) {
                auto it_end = anoc->ChildList_End();
                for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
                    AbstractNamedObject::ptr_type ano = *it;
                    AbstractNamedObjectContainer::ptr_type anoc =
                        std::dynamic_pointer_cast<AbstractNamedObjectContainer>(ano);
                    if (anoc) {
                        anoStack.push_back(anoc);
                    } else {
                        core::CallerSlot* callerSlot = dynamic_cast<core::CallerSlot*>((*it).get());
                        if (callerSlot != nullptr) {
                            core::Call* call = callerSlot->CallAs<Call>();
                            if (call != nullptr) {
                                auto target = call->PeekCalleeSlot();
                                auto source = call->PeekCallerSlot();
                                if (source->FullName().Equals(cdr.First()) && target->FullName().Equals(cdr.Second())) {
                                    // this should be the right call
                                    // vislib::sys::Log::DefaultLog.WriteInfo("found call from %s to %s",
                                    // call->PeekCallerSlot()->FullName(),
                                    //    call->PeekCalleeSlot()->FullName());
                                    callerSlot->SetCleanupMark(true);
                                    callerSlot->DisconnectCalls();
                                    callerSlot->PerformCleanup();
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!found) {
            vislib::sys::Log::DefaultLog.WriteError(
                "cannot delete call from \"%s\" to \"%s\"", cdr.First().PeekBuffer(), cdr.Second().PeekBuffer());
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingModuleInstRequests.Count(), this->moduleInstRequestsFlushIndices);

    // make modules
    while (this->pendingModuleInstRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->moduleInstRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->moduleInstRequestsFlushIndices);
            break;
        }

        auto mir = this->pendingModuleInstRequests.First();
        this->pendingModuleInstRequests.RemoveFirst();

        ++counter;

        if (this->instantiateModule(mir.First(), mir.Second()) == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("cannot instantiate module \"%s\""
                                                    " of class \"%s\".",
                mir.First().PeekBuffer(), mir.Second()->ClassName());
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingCallInstRequests.Count(), this->callInstRequestsFlushIndices);

    // make calls
    while (this->pendingCallInstRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->callInstRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->callInstRequestsFlushIndices);
            break;
        }

        auto cir = this->pendingCallInstRequests.First();
        this->pendingCallInstRequests.RemoveFirst();

        ++counter;

        if (this->InstantiateCall(cir.From(), cir.To(), cir.Description()) == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("cannot instantiate \"%s\" call"
                                                    " from \"%s\" to \"%s\".",
                cir.Description()->ClassName(), cir.From().PeekBuffer(), cir.To().PeekBuffer());
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingChainCallInstRequests.Count(), this->chainCallInstRequestsFlushIndices);

    // make chain calls
    while (this->pendingChainCallInstRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->chainCallInstRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->chainCallInstRequestsFlushIndices);
            break;
        }

        auto cir = this->pendingChainCallInstRequests.First();
        this->pendingChainCallInstRequests.RemoveFirst();

        ++counter;

        ASSERT(cir.From().StartsWith("::"));

        std::string chainClass = cir.Description()->ClassName();

        vislib::Array<vislib::StringA> fromDirs = vislib::StringTokeniserA::Split(cir.From(), "::", true);
        vislib::StringA fromSlotName = fromDirs.Last();
        fromDirs.RemoveLast();
        vislib::StringA fromModName = fromDirs.Last();
        fromDirs.RemoveLast();

        ModuleNamespace::ptr_type fromNS =
            ModuleNamespace::dynamic_pointer_cast(this->namespaceRoot->FindNamespace(fromDirs, false));
        if (!fromNS) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to instantiate call: can not find source namespace \"%s\"", cir.From().PeekBuffer());
            continue;
        }

        Module::ptr_type fromMod = std::dynamic_pointer_cast<Module>(fromNS->FindChild(fromModName));
        if (!fromMod) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to instantiate call: can not find source module \"%s\"", cir.From().PeekBuffer());
            continue;
        }

        CallerSlot* fromSlot = dynamic_cast<CallerSlot*>(fromMod->FindSlot(fromSlotName));
        if (fromSlot == NULL) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to instantiate call: can not find source slot \"%s\"", cir.From().PeekBuffer());
            continue;
        }

        std::vector<AbstractNamedObject::const_ptr_type> anoStack;

        const auto it_end = this->namespaceRoot->ChildList_End();
        for (auto it = this->namespaceRoot->ChildList_Begin(); it != it_end; ++it) {
            if (std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(*it)) {
                anoStack.push_back(*it);
            }
        }

        typedef struct {
            std::string name;
            std::string fromFull;
            std::string toFull;
        } conn;
        std::vector<conn> connections;

        while (!anoStack.empty()) {
            AbstractNamedObject::const_ptr_type ano = anoStack.back();
            anoStack.pop_back();

            AbstractNamedObjectContainer::const_ptr_type anoc =
                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
            const CallerSlot* caller = dynamic_cast<const CallerSlot*>(ano.get());

            if (caller) {
                // TODO there must be a better way
                const Call* c = const_cast<CallerSlot*>(caller)->CallAs<Call>();
                if (c != nullptr) {
                    if (c->ClassName() == chainClass) {
                        conn theConn = {c->ClassName(), c->PeekCallerSlot()->FullName().PeekBuffer(),
                            c->PeekCalleeSlot()->FullName().PeekBuffer()};
                        connections.push_back(theConn);
                    }
                    // answer << c->ClassName() << ";" << c->PeekCallerSlot()->Parent()->Name() << ","
                    //       << c->PeekCalleeSlot()->Parent()->Name() << ";" << c->PeekCallerSlot()->Name() << ","
                    //       << c->PeekCalleeSlot()->Name() << std::endl;
                }
            }

            if (anoc) {
                const auto it_end2 = anoc->ChildList_End();
                for (auto it = anoc->ChildList_Begin(); it != it_end2; ++it) {
                    anoStack.push_back(*it);
                }
            }
        }

        // for (auto& conn : connections) {
        //    vislib::sys::Log::DefaultLog.WriteInfo(
        //        "connection: %s --%s-> %s", conn.fromFull.c_str(), conn.name.c_str(), conn.toFull.c_str());
        //}

        vislib::sys::Log::DefaultLog.WriteInfo("chain call: trying to find slot for appending %s at the end of %s",
            cir.Description()->ClassName(), cir.From().PeekBuffer());
        std::string currFrom = cir.From().PeekBuffer();

        std::string currTo = "";
        std::string modFull;
        // find the one connection going out from the chain start
        auto which = std::find_if(
            connections.begin(), connections.end(), [currFrom](conn& c) { return c.fromFull == currFrom; });
        if (which == connections.end()) {
            // nothing found, this is the same as a normal call connection
            vislib::sys::Log::DefaultLog.WriteInfo(
                "chain call: there is nothing connected to %s, appending %s directly", cir.From().PeekBuffer(),
                cir.Description()->ClassName());
            if (this->InstantiateCall(cir.From(), cir.To(), cir.Description()) == nullptr) {
                vislib::sys::Log::DefaultLog.WriteError("chain call: cannot instantiate \"%s\" call"
                                                        " from \"%s\" to \"%s\".",
                    cir.Description()->ClassName(), cir.From().PeekBuffer(), cir.To().PeekBuffer());
            }
        } else {
            // walk the connection chain until we find an end

            // step out of the starting module.
            vislib::sys::Log::DefaultLog.WriteInfo(
                "chain call: following chain from %s to %s", which->fromFull.c_str(), which->toFull.c_str());
            auto pos = which->toFull.find_last_of("::");
            modFull = which->toFull.substr(0, pos - 1);

            std::vector<std::string> candidateModules;
            candidateModules.push_back(modFull);

            while (!candidateModules.empty()) {
                auto cand = candidateModules.back();
                candidateModules.pop_back();

                Module::ptr_type mod =
                    Module::dynamic_pointer_cast(this->namespaceRoot.get()->FindNamedObject(cand.c_str()));
                if (!mod) {
                    vislib::sys::Log::DefaultLog.WriteError("chain call: cannot get module %s", cand.c_str());
                    continue;
                }

                AbstractNamedObjectContainer::child_list_type::iterator si, se;
                se = mod->ChildList_End();
                for (si = mod->ChildList_Begin(); si != se; ++si) {
                    CallerSlot* slot = dynamic_cast<CallerSlot*>((*si).get());
                    if (slot) {
                        if (slot->IsCallCompatible(cir.Description())) {

                            auto connectedCall = std::find_if(connections.begin(), connections.end(),
                                [slot](conn& currConn) { return currConn.fromFull == std::string(slot->FullName()); });
                            if (connectedCall != connections.end()) {
                                auto pos = connectedCall->toFull.find_last_of("::");
                                candidateModules.push_back(connectedCall->toFull.substr(0, pos - 1));
                                vislib::sys::Log::DefaultLog.WriteInfo("chaincall: %s is connected to %s, pushing.",
                                    slot->FullName().PeekBuffer(), connectedCall->toFull.c_str());
                            } else {
                                vislib::sys::Log::DefaultLog.WriteInfo(
                                    "chain connection from slot %s", slot->FullName().PeekBuffer());
                                if (this->InstantiateCall(slot->FullName(), cir.To(), cir.Description()) == nullptr) {
                                    vislib::sys::Log::DefaultLog.WriteError("cannot instantiate \"%s\" call"
                                                                            " from \"%s\" to \"%s\".",
                                        cir.Description()->ClassName(), slot->FullName().PeekBuffer(),
                                        cir.To().PeekBuffer());
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingParamSetRequests.Count(), this->paramSetRequestsFlushIndices);

    // set parameter values;
    while (this->pendingParamSetRequests.Count() > 0) {
        // flush mechanism
        if (this->checkForFlushEvent(counter, this->paramSetRequestsFlushIndices)) {
            this->updateFlushIdxList(counter, this->paramSetRequestsFlushIndices);
            break;
        }

        auto psr = this->pendingParamSetRequests.First();
        this->pendingParamSetRequests.RemoveFirst();

        ++counter;

        auto p = this->FindParameter(psr.First());
        if (p != nullptr) {
            vislib::TString val;
            vislib::UTF8Encoder::Decode(val, psr.Second());

            if (!p->ParseValue(val)) {
                vislib::sys::Log::DefaultLog.WriteError("Setting parameter \"%s\" to \"%s\": ParseValue failed.",
                    psr.First().PeekBuffer(), psr.Second().PeekBuffer());
                continue;
            } else {
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Setting parameter \"%s\" to \"%s\".", psr.First().PeekBuffer(), psr.Second().PeekBuffer());
            }
        } else {
            // the error is already shown
            continue;
        }
    }

    counter = 0;

    this->shortenFlushIdxList(this->pendingGroupParamSetRequests.Count(), this->groupParamSetRequestsFlushIndices);

    // set parameter values for a group
    auto pgp = this->pendingGroupParamSetRequests.GetIterator();
    while (pgp.HasNext()) {
        auto& pg = pgp.Next().Value();
        if (pg.GroupSize == pg.Requests.Count()) {
            // flush mechanism
            if (this->checkForFlushEvent(
                    counter, this->groupParamSetRequestsFlushIndices)) { // TODO Is this the right place?
                this->updateFlushIdxList(counter, this->groupParamSetRequestsFlushIndices);
                break;
            }

            ++counter;

            vislib::sys::Log::DefaultLog.WriteInfo("parameter group %s is complete. executing:", pg.Name.PeekBuffer());
            auto pgi = pg.Requests.GetIterator();
            while (pgi.HasNext()) {
                auto& pr = pgi.Next();

                auto p = this->FindParameter(pr.Key());
                if (p != nullptr) {
                    vislib::TString val;
                    vislib::UTF8Encoder::Decode(val, pr.Value());

                    if (!p->ParseValue(val)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Setting parameter \"%s\" to \"%s\": ParseValue failed.", pr.Key().PeekBuffer(),
                            pr.Value().PeekBuffer());
                        continue;
                    } else {
                        vislib::sys::Log::DefaultLog.WriteInfo(
                            "Setting parameter \"%s\" to \"%s\".", pr.Key().PeekBuffer(), pr.Value().PeekBuffer());
                    }
                } else {
                    // the error is already shown
                    continue;
                }
            }
            pg.Requests.Clear();
        }
    }
}


/*
 * megamol::core::CoreInstance::GetPendingViewName
 */
vislib::StringA megamol::core::CoreInstance::GetPendingViewName(void) {
    vislib::sys::AutoLock l(this->graphUpdateLock);
    if (this->pendingViewInstRequests.IsEmpty()) return nullptr;
    ViewInstanceRequest request = this->pendingViewInstRequests.First();
    return request.Name();
}


/*
 * megamol::core::CoreInstance::InstantiatePendingView
 */
megamol::core::ViewInstance::ptr_type megamol::core::CoreInstance::InstantiatePendingView(void) {
    using vislib::sys::Log;

    vislib::sys::AutoLock l(this->graphUpdateLock);
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    if (this->pendingViewInstRequests.IsEmpty()) return NULL;

    ViewInstanceRequest request = this->pendingViewInstRequests.First();
    this->pendingViewInstRequests.RemoveFirst();

    std::shared_ptr<ModuleNamespace> preViewInst = NULL;
    bool hasErrors = false;
    view::AbstractView *view = NULL, *fallbackView = NULL;
    vislib::StringA viewFullPath =
        this->namespaceRoot->FullNamespace(request.Name(), request.Description()->ViewModuleID());

    AbstractNamedObject::ptr_type ano = this->namespaceRoot->FindChild(request.Name());
    if (ano) {
        preViewInst = std::dynamic_pointer_cast<ModuleNamespace>(ano);
        if (!preViewInst) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate view %s: non-namespace object blocking instance name\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    } else {
        preViewInst = std::make_shared<ModuleNamespace>(request.Name());
        this->namespaceRoot->AddChild(preViewInst);
    }

    if (!preViewInst) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to instantiate view %s: Internal Error %d\n",
            request.Name().PeekBuffer(), __LINE__);
        return NULL;
    }

    // instantiate modules
    for (unsigned int idx = 0; idx < request.Description()->ModuleCount(); idx++) {
        const ViewDescription::ModuleInstanceRequest& mir = request.Description()->Module(idx);
        factories::ModuleDescription::ptr desc = mir.Second();

        vislib::StringA fullName = this->namespaceRoot->FullNamespace(request.Name(), mir.First());

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n",
                fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module::ptr_type mod = this->instantiateModule(fullName, desc);
        if (!mod) {
            hasErrors = true;
            continue;

        } else {
            view::AbstractView* av = dynamic_cast<view::AbstractView*>(mod.get());
            if (av != NULL) {
                // view module instantiated.
                if (fullName.Equals(viewFullPath)) {
                    view = av;
                } else if (fallbackView == NULL) {
                    fallbackView = av;
                }
            }
        }
    }

    if (view == NULL) {
        if (fallbackView != NULL) {
            view = fallbackView;
        } else {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to instantiate view %s: No view module found\n", request.Name().PeekBuffer());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < request.Description()->CallCount(); idx++) {
        const ViewDescription::CallInstanceRequest& cir = request.Description()->Call(idx);
        factories::CallDescription::ptr desc = cir.Description();

        vislib::StringA fromFullName = this->namespaceRoot->FullNamespace(request.Name(), cir.From());
        vislib::StringA toFullName = this->namespaceRoot->FullNamespace(request.Name(), cir.To());

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call* call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL))
                profiler::Manager::Instance().Select(fromFullName);
        }
    }

    if (hasErrors) {
        this->CleanupModuleGraph();

    } else {
        // Create Instance object replacing the temporary namespace
        ViewInstance::ptr_type inst(new ViewInstance());
        if (!inst) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to construct instance %s\n", request.Name().PeekBuffer());
            return NULL;
        }

        if (!dynamic_cast<ViewInstance*>(inst.get())->Initialize(preViewInst, view)) {
            inst.reset();
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to initialize instance %s\n", request.Name().PeekBuffer());
            return NULL;
        }

        this->applyConfigParams(request.Name(), request.Description(), &request);

        return inst;
    }

    return NULL;
}


/*
 * megamol::core::CoreInstance::instantiateSubView
 */
megamol::core::view::AbstractView* megamol::core::CoreInstance::instantiateSubView(megamol::core::ViewDescription* vd) {
    using vislib::sys::Log;
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    bool hasErrors = false;
    view::AbstractView *view = NULL, *fallbackView = NULL;

    // instantiate modules
    for (unsigned int idx = 0; idx < vd->ModuleCount(); idx++) {
        const ViewDescription::ModuleInstanceRequest& mir = vd->Module(idx);
        factories::ModuleDescription::ptr desc = mir.Second();
        const vislib::StringA& fullName = mir.First();

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n",
                fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module::ptr_type mod = this->instantiateModule(fullName, desc);
        if (mod == NULL) {
            hasErrors = true;
            continue;

        } else {
            view::AbstractView* av = dynamic_cast<view::AbstractView*>(mod.get());
            if (av != NULL) {
                // view module instantiated.
                if (fullName.Equals(vd->ViewModuleID())) {
                    view = av;
                } else if (fallbackView == NULL) {
                    fallbackView = av;
                }
            }
        }
    }

    if (view == NULL) {
        if (fallbackView != NULL) {
            view = fallbackView;
        } else {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to instantiate view %s: No view module found\n", vd->ClassName());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < vd->CallCount(); idx++) {
        const ViewDescription::CallInstanceRequest& cir = vd->Call(idx);
        factories::CallDescription::ptr desc = cir.Description();
        const vislib::StringA& fromFullName = cir.From();
        const vislib::StringA& toFullName = cir.To();

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call* call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL))
                profiler::Manager::Instance().Select(fromFullName);
        }
    }

    if (hasErrors) {
        view = NULL;
        this->CleanupModuleGraph();
    } else {
        this->applyConfigParams("", vd, vd);
    }

    return view;
}


/*
 * megamol::core::CoreInstance::InstantiatePendingJob
 */
megamol::core::JobInstance::ptr_type megamol::core::CoreInstance::InstantiatePendingJob(void) {
    using vislib::sys::Log;
    vislib::sys::AutoLock l(this->graphUpdateLock);
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    if (this->pendingJobInstRequests.IsEmpty()) return NULL;

    JobInstanceRequest request = this->pendingJobInstRequests.First();
    this->pendingJobInstRequests.RemoveFirst();

    ModuleNamespace::ptr_type preJobInst;
    bool hasErrors = false;
    job::AbstractJob *job = NULL, *fallbackJob = NULL;
    vislib::StringA jobFullPath =
        this->namespaceRoot->FullNamespace(request.Name(), request.Description()->JobModuleID());

    AbstractNamedObject::ptr_type ano = this->namespaceRoot->FindChild(request.Name());
    if (ano) {
        preJobInst = ModuleNamespace::dynamic_pointer_cast(ano);
        if (!preJobInst) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate job %s: non-namespace object blocking instance name\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    } else {
        preJobInst = std::make_shared<ModuleNamespace>(request.Name());
        this->namespaceRoot->AddChild(preJobInst);
    }

    if (!preJobInst) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to instantiate job %s: Internal Error %d\n",
            request.Name().PeekBuffer(), __LINE__);
        return NULL;
    }

    // instantiate modules
    for (unsigned int idx = 0; idx < request.Description()->ModuleCount(); idx++) {
        const JobDescription::ModuleInstanceRequest& mir = request.Description()->Module(idx);
        factories::ModuleDescription::ptr desc = mir.Second();

        vislib::StringA fullName = this->namespaceRoot->FullNamespace(request.Name(), mir.First());

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n",
                fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module::ptr_type mod = this->instantiateModule(fullName, desc);
        if (!mod) {
            hasErrors = true;
            continue;

        } else {
            job::AbstractJob* aj = dynamic_cast<job::AbstractJob*>(mod.get());
            if (aj != NULL) {
                // view module instantiated.
                if (fullName.Equals(jobFullPath)) {
                    job = aj;
                } else if (fallbackJob == NULL) {
                    fallbackJob = aj;
                }
            }
        }
    }

    if (job == NULL) {
        if (fallbackJob != NULL) {
            job = fallbackJob;
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to instantiate job %s: No job controller module found\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < request.Description()->CallCount(); idx++) {
        const JobDescription::CallInstanceRequest& cir = request.Description()->Call(idx);
        factories::CallDescription::ptr desc = cir.Description();

        vislib::StringA fromFullName = this->namespaceRoot->FullNamespace(request.Name(), cir.From());
        vislib::StringA toFullName = this->namespaceRoot->FullNamespace(request.Name(), cir.To());

        if (!desc) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call* call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL))
                profiler::Manager::Instance().Select(fromFullName);
        }
    }

    if (hasErrors) {
        this->CleanupModuleGraph();

    } else {
        // Create Instance object replacing the temporary namespace
        JobInstance::ptr_type inst = std::make_shared<JobInstance>();
        if (inst == NULL) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to construct instance %s\n", request.Name().PeekBuffer());
            return NULL;
        }

        if (!dynamic_cast<JobInstance*>(inst.get())->Initialize(preJobInst, job)) {
            inst.reset();
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to initialize instance %s\n", request.Name().PeekBuffer());
            return NULL;
        }

        this->applyConfigParams(request.Name(), request.Description(), &request);

        return inst;
    }

    return NULL;
}

/*
 * megamol::core::CoreInstance::FindParameter
 */
vislib::SmartPtr<megamol::core::param::AbstractParam> megamol::core::CoreInstance::FindParameterIndirect(
    const vislib::StringA& name, bool quiet) {
    vislib::StringA paramName(name);
    vislib::SmartPtr<core::param::AbstractParam> param;
    vislib::SmartPtr<core::param::AbstractParam> lastParam;
    while ((param = this->FindParameter(paramName, quiet)) != nullptr) {
        lastParam = param;
        paramName = param->ValueString();
    }
    return lastParam;
}

/*
 * megamol::core::CoreInstance::FindParameter
 */
vislib::SmartPtr<megamol::core::param::AbstractParam> megamol::core::CoreInstance::FindParameter(
    const vislib::StringA& name, bool quiet, bool create) {
    using vislib::sys::Log;
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
                /*	if(create)
                    {
                        param::ParamSlot *slotNew = new param::ParamSlot(name, "newly inserted");
                        *slotNew << new param::StringParam("");
                        slotNew->MakeAvailable();
                        this->namespaceRoot.AddChild(slotNew);
                    }
                    else*/
                {
                    if (!quiet)
                        Log::DefaultLog.WriteMsg(
                            Log::LEVEL_ERROR, "Cannot find parameter \"%s\": namespace not found", name.PeekBuffer());
                    return NULL;
                }
            }
        }
    }

    Module::ptr_type mod = Module::dynamic_pointer_cast(mn->FindChild(modName));
    if (!mod) {
        /*	if(create)
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
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Cannot find parameter \"%s\": module not found", name.PeekBuffer());
            return NULL;
        }
    }

    param::ParamSlot* slot = dynamic_cast<param::ParamSlot*>(mod->FindChild(slotName).get());
    if (slot == NULL) {
        /*	if(create)
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
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot not found", name.PeekBuffer());
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
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot is not available", name.PeekBuffer());
            return NULL;
        }
    }
    if (slot->Parameter().IsNull()) {
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
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot has no parameter", name.PeekBuffer());
            return NULL;
        }
    }


    return slot->Parameter();
}


std::string megamol::core::CoreInstance::GetProjectFromPNG(std::string filename) {
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        vislib::sys::Log::DefaultLog.WriteError("getProjectFromPNG: Unable to create png struct");
    } else {
        FILE* fp = fopen(filename.c_str(), "rb");
        if (fp == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("getProjectFromPNG: Unable to open png file \"%s\"", filename.c_str());
        } else {
            png_infop info = png_create_info_struct(png);
            if (!info) {
                vislib::sys::Log::DefaultLog.WriteError("getProjectFromPNG: Unable to create png info struct");
            } else {
                setjmp(png_jmpbuf(png));
                png_init_io(png, fp);
                png_read_info(png, info);
                png_uint_32 exif_size = 0;
                png_bytep exif_data = nullptr;
                png_get_eXIf_1(png, info, &exif_size, &exif_data);
                if (exif_size > 0) {
                    std::string content(reinterpret_cast<char*>(exif_data));
                    return content;
                } else {
                    vislib::sys::Log::DefaultLog.WriteError("LoadProject: Unable to extract png exif data");
                }
                png_destroy_info_struct(png, &info);
            }
            fclose(fp);
        }
        png_destroy_read_struct(&png, nullptr, nullptr);
        // exif_data buffer seems to live inside exif_info and is disposed automatically
    }
    return "";
}

/*
 * megamol::core::CoreInstance::LoadProject
 */
void megamol::core::CoreInstance::LoadProject(const vislib::StringA& filename) {
    if (filename.EndsWith(".lua")) {
        vislib::StringA content;
        std::string result;
        if (!vislib::sys::ReadTextFile(content, filename)) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to open project file \"%s\"", filename.PeekBuffer());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO, "Loading project file \"%s\"", filename.PeekBuffer());
            if (!this->lua->RunString(content.PeekBuffer(), result, filename.PeekBuffer())) {
                vislib::sys::Log::DefaultLog.WriteError(vislib::sys::Log::LEVEL_INFO,
                    "Failed loading project file \"%s\": %s", filename.PeekBuffer(), result.c_str());
            } else {
                this->loadedLuaProjects.Add(vislib::Pair<vislib::StringA, vislib::StringA>(filename, content));
            }
        }
    } else if (filename.EndsWith(".png")) {
        std::string result;
        std::string content = GetProjectFromPNG(filename.PeekBuffer());
        //vislib::sys::Log::DefaultLog.WriteInfo("Loaded project from png:\n%s", content.c_str());
        if (!this->lua->RunString(content.c_str(), result, filename.PeekBuffer())) {
            vislib::sys::Log::DefaultLog.WriteError(vislib::sys::Log::LEVEL_INFO,
                "Failed loading project file \"%s\": %s", filename.PeekBuffer(), result.c_str());
        } else {
            this->loadedLuaProjects.Add(vislib::Pair<vislib::StringA, vislib::StringA>(filename, content.c_str()));
        }
    } else {
        megamol::core::utility::xml::XmlReader reader;
        if (!reader.OpenFile(filename)) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to open project file \"%s\"", filename.PeekBuffer());
            return;
        }
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO, "Loading project file \"%s\"", filename.PeekBuffer());
        this->addProject(reader);
    }
}


/*
 * megamol::core::CoreInstance::LoadProject
 */
void megamol::core::CoreInstance::LoadProject(const vislib::StringW& filename) {
    if (filename.EndsWith(L".lua")) {
        vislib::StringA content;
        std::string result;
        if (!vislib::sys::ReadTextFile(content, filename)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to open project file \"%s\"",
                vislib::StringA(filename).PeekBuffer());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO, "Loading project file \"%s\"", vislib::StringA(filename).PeekBuffer());
            if (!this->lua->RunString(content.PeekBuffer(), result)) {
                vislib::sys::Log::DefaultLog.WriteError(vislib::sys::Log::LEVEL_INFO,
                    "Failed loading project file \"%s\": %s", vislib::StringA(filename).PeekBuffer(), result.c_str());
            } else {
                this->loadedLuaProjects.Add(
                    vislib::Pair<vislib::StringA, vislib::StringA>(vislib::StringA(filename), content));
            }
        }
    } else if (filename.EndsWith(L".png")) {
        std::string result;
        std::string content = GetProjectFromPNG(W2A(filename.PeekBuffer()));
        // vislib::sys::Log::DefaultLog.WriteInfo("Loaded project from png:\n%s", content.c_str());
        if (!this->lua->RunString(content.c_str(), result, W2A(filename.PeekBuffer()))) {
            vislib::sys::Log::DefaultLog.WriteError(vislib::sys::Log::LEVEL_INFO,
                "Failed loading project file \"%s\": %s", filename.PeekBuffer(), result.c_str());
        } else {
            this->loadedLuaProjects.Add(vislib::Pair<vislib::StringA, vislib::StringA>(filename, content.c_str()));
        }
    } else {
        megamol::core::utility::xml::XmlReader reader;
        if (!reader.OpenFile(filename)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to open project file \"%s\"",
                vislib::StringA(filename).PeekBuffer());
            return;
        }
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO, "Loading project file \"%s\"", vislib::StringA(filename).PeekBuffer());
        this->addProject(reader);
    }
}


void megamol::core::CoreInstance::SerializeGraph(std::string& serInstances, std::string& serModules, std::string& serCalls, std::string& serParams) {

    std::stringstream confInstances, confModules, confCalls, confParams;

    std::map<std::string, std::string> view_instances;
    std::map<std::string, std::string> job_instances;
    {
        vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());
        AbstractNamedObjectContainer::ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        int job_counter = 0;
        for (auto ano = anoc->ChildList_Begin(); ano != anoc->ChildList_End(); ++ano) {
            auto vi = dynamic_cast<ViewInstance*>(ano->get());
            auto ji = dynamic_cast<JobInstance*>(ano->get());
            if (vi && vi->View()) {
                std::string vin = vi->Name().PeekBuffer();
                view_instances[vi->View()->FullName().PeekBuffer()] = vin;
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "ScreenShooter: found view instance \"%s\" with view \"%s\".",
                    view_instances[vi->View()->FullName().PeekBuffer()].c_str(),
                    vi->View()->FullName().PeekBuffer());
            }
            if (ji && ji->Job()) {
                std::string jin = ji->Name().PeekBuffer();
                // todo: find job module! WTF!
                job_instances[jin] = std::string("job") + std::to_string(job_counter);
                vislib::sys::Log::DefaultLog.WriteInfo("ScreenShooter: found job instance \"%s\" with job \"%s\".",
                    jin.c_str(), job_instances[jin].c_str());
                ++job_counter;
            }
        }

        const auto fun = [&confInstances, &confModules, &confCalls, &confParams, &view_instances](Module* mod) {
            if (view_instances.find(mod->FullName().PeekBuffer()) != view_instances.end()) {
                confInstances << "mmCreateView(\"" << view_instances[mod->FullName().PeekBuffer()] << "\",\""
                    << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer() << "\")\n";
            }
            else {
                // todo: jobs??
                confModules << "mmCreateModule(\"" << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer()
                    << "\")\n";
            }
            AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin();
                si != se; ++si) {
                const auto slot = dynamic_cast<param::ParamSlot*>((*si).get());
                if (slot) {
                    const auto bp = slot->Param<param::ButtonParam>();
                    if (!bp) {
                        std::string val = slot->Parameter()->ValueString().PeekBuffer();
                        // caution: value strings could contain unescaped quotes, so fix that:
                        //std::string from = "\"";
                        //std::string to = "\\\"";
                        //size_t start_pos = 0;
                        //while ((start_pos = val.find(from, start_pos)) != std::string::npos) {
                        //    val.replace(start_pos, from.length(), to);
                        //    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
                        //}
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


/*
 * megamol::core::CoreInstance::GetGlobalParameterHash
 */
size_t megamol::core::CoreInstance::GetGlobalParameterHash(void) {

    ParamHashMap_t current_map;
    this->getGlobalParameterHash(this->namespaceRoot, current_map);
    if (!mapCompare(current_map, this->lastParamMap)) {
        this->lastParamMap = current_map;
        this->parameterHash++;
    }

    return this->parameterHash;
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
 * megamol::core::CoreInstance::getGlobalParameterHash
 */
void megamol::core::CoreInstance::getGlobalParameterHash(
    megamol::core::ModuleNamespace::const_ptr_type path, ParamHashMap_t& map) const {

    AbstractNamedObjectContainer::child_list_type::const_iterator i, e;
    e = path->ChildList_End();
    for (i = path->ChildList_Begin(); i != e; ++i) {
        AbstractNamedObject::const_ptr_type child = *i;
        Module::const_ptr_type mod = Module::dynamic_pointer_cast(child);
        ModuleNamespace::const_ptr_type ns = ModuleNamespace::dynamic_pointer_cast(child);

        if (mod) {
            AbstractNamedObjectContainer::child_list_type::const_iterator si, se;
            se = mod->ChildList_End();
            for (si = mod->ChildList_Begin(); si != se; ++si) {
                const param::ParamSlot* slot = dynamic_cast<const param::ParamSlot*>((*si).get());
                if (slot != NULL) {
                    std::string name(mod->FullName());
                    // vislib::StringA name(mod->FullName());
                    name.append("::");
                    name.append(slot->Name());
                    map.emplace(std::make_pair(name, slot->Parameter()->GetHash()));
                }
            }

        } else if (ns) {
            this->getGlobalParameterHash(ns, map);
        }
    }
}


/*
 * megamol::core::CoreInstance::GetInstanceTime
 */
double megamol::core::CoreInstance::GetCoreInstanceTime(void) const {
    //#ifdef _WIN32 // not a good idea when running more than one megamol on one machine
    //    ::SetThreadAffinityMask(::GetCurrentThread(), 0x00000001);
    //#endif /* _WIN32 */
    return vislib::sys::PerformanceCounter::QueryMillis() * 0.001 + this->timeOffset;
}


/*
 * megamol::core::CoreInstance::OffsetInstanceTime
 */
void megamol::core::CoreInstance::OffsetInstanceTime(double offset) {
    this->timeOffset += offset;
    //    printf("------------------------------------------------------------ %f\n", this->timeOffset);
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
 * megamol::core::CoreInstance::SetupGraphFromNetwork
 */
void megamol::core::CoreInstance::SetupGraphFromNetwork(const void* data) {
    using vislib::net::AbstractSimpleMessage;
    using vislib::sys::Log;

    const AbstractSimpleMessage* dataPtr = static_cast<const AbstractSimpleMessage*>(data);
    const AbstractSimpleMessage& dat = *dataPtr;

    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());
    try {

        UINT64 cntMods = *dat.GetBodyAs<UINT64>();
        UINT64 cntCalls = *dat.GetBodyAsAt<UINT64>(sizeof(UINT64));
        UINT64 cntParams = *dat.GetBodyAsAt<UINT64>(sizeof(UINT64) * 2);
        SIZE_T pos = sizeof(UINT64) * 3;

        // printf("\n\nGraph Setup:\n");
        for (UINT64 i = 0; i < cntMods; i++) {
            vislib::StringA modClass(dat.GetBodyAsAt<char>(pos));
            pos += modClass.Length() + 1;
            vislib::StringA modName(dat.GetBodyAsAt<char>(pos));
            pos += modName.Length() + 1;

            if (modClass.Equals(cluster::ClusterViewMaster::ClassName()) ||
                modClass.Equals(cluster::ClusterController::ClassName()) ||
                modClass.Equals(cluster::simple::Server::ClassName())) {
                // these are infra structure modules and not to be synced
                continue;
            }

            factories::ModuleDescription::ptr d = this->GetModuleDescriptionManager().Find(modClass);
            if (!d) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s(%s): class description not found\n", modName.PeekBuffer(),
                    modClass.PeekBuffer());
                continue;
            }
            Module::ptr_type m = this->instantiateModule(modName, d);
            if (!m) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s(%s): instantiation failed\n", modName.PeekBuffer(),
                    modClass.PeekBuffer());
                continue;
            }
            // printf("    Modul: %s as %s\n", modClass.PeekBuffer(), modName.PeekBuffer());
        }

        for (UINT64 i = 0; i < cntCalls; i++) {
            vislib::StringA callClass(dat.GetBodyAsAt<char>(pos));
            pos += callClass.Length() + 1;
            vislib::StringA callFrom(dat.GetBodyAsAt<char>(pos));
            pos += callFrom.Length() + 1;
            vislib::StringA callTo(dat.GetBodyAsAt<char>(pos));
            pos += callTo.Length() + 1;

            AbstractNamedObject::ptr_type ano = this->namespaceRoot->FindNamedObject(callFrom);
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Not connecting call %s from missing module %s\n", callClass.PeekBuffer(), callFrom.PeekBuffer());
                continue;
            }
            ano = this->namespaceRoot->FindNamedObject(callTo);
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Not connecting call %s to missing module %s\n", callClass.PeekBuffer(), callTo.PeekBuffer());
                continue;
            }

            factories::CallDescription::ptr d = this->GetCallDescriptionManager().Find(callClass);
            if (!d) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate call %s=>%s(%s): class description not found\n", callFrom.PeekBuffer(),
                    callTo.PeekBuffer(), callClass.PeekBuffer());
                continue;
            }
            Call* c = this->InstantiateCall(callFrom, callTo, d);
            if (c == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s=>%s(%s): instantiation failed\n", callFrom.PeekBuffer(),
                    callTo.PeekBuffer(), callClass.PeekBuffer());
                continue;
            }
            // printf("    Call: %s from %s to %s\n", callClass.PeekBuffer(), callFrom.PeekBuffer(),
            // callTo.PeekBuffer());
        }

        for (UINT64 i = 0; i < cntParams; i++) {
            vislib::StringA paramName(dat.GetBodyAsAt<char>(pos));
            pos += paramName.Length() + 1;
            vislib::StringA paramValue(dat.GetBodyAsAt<char>(pos));
            pos += paramValue.Length() + 1;

            AbstractNamedObject::ptr_type ano = this->namespaceRoot->FindNamedObject(paramName);
            if (!ano) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Parameter %s not found\n", paramName.PeekBuffer());
                continue;
            }
            param::ParamSlot* ps = dynamic_cast<param::ParamSlot*>(ano.get());
            if (ps == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s is not a parameter slot\n", paramName.PeekBuffer());
                continue;
            }

            if (ps->Parameter().IsNull()) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Parameter of %s is NULL\n", paramName.PeekBuffer());
                continue;
            }

            vislib::TString value;
            if (!vislib::UTF8Encoder::Decode(value, paramValue)) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_WARN, "Unable to decode parameter value for %s\n", paramName.PeekBuffer());
                continue;
            }
            ps->Parameter()->ParseValue(value);
            // printf("    Param: %s to %s\n", paramName.PeekBuffer(), paramValue.PeekBuffer());
        }
        // printf("\n");

    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to setup module graph from network message: %s\n", ex.GetMsgA());
    } catch (...) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to setup module graph from network message: unexpected exception\n");
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
}


/*
 * megamol::core::CoreInstance::Quickstart
 */
void megamol::core::CoreInstance::Quickstart(const vislib::TString& filename) {
    using vislib::sys::Log;
    Log::DefaultLog.WriteInfo(_T("Quickstarting \"%s\" ..."), filename.PeekBuffer());

    const SIZE_T maxBufferSize = 4 * 1024;
    unsigned char* buffer = new unsigned char[maxBufferSize];

    vislib::sys::File file;
    if (!file.Open(
            filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        delete[] buffer;
        Log::DefaultLog.WriteError(_T("Failed to Quickstart \"%s\": Unable to load peek data"), filename.PeekBuffer());
        return;
    }
    SIZE_T bufferSize = static_cast<SIZE_T>(file.Read(buffer, maxBufferSize));
    file.Close();

    if (bufferSize <= 0) {
        delete[] buffer;
        Log::DefaultLog.WriteError(_T("Failed to Quickstart \"%s\": Unable to load peek data"), filename.PeekBuffer());
        return;
    }

    // auto-detect data source module
    factories::ModuleDescription::ptr dataSrcClass = NULL;
    //    factories::ModuleDescriptionManager::description_const_iterator_type di =
    //    ModuleDescriptionManager::Instance()->GetIterator();

    // first try auto-detect with loaders with matching file name extension
    //    while (di.HasNext()) {
    //        ModuleDescription *md = di.Next();
    for (auto md : this->GetModuleDescriptionManager()) {
        if (!md->IsLoaderWithAutoDetection()) continue;
        const char* extsStr = md->LoaderAutoDetectionFilenameExtensions();
        if (extsStr == NULL) continue;
        vislib::Array<vislib::StringA> exts = vislib::StringTokeniserA::Split(extsStr, ';', true);
        bool found = false;
        for (SIZE_T i = 0; i < exts.Count(); i++) {
            if (filename.EndsWith(A2T(exts[i]))) {
                found = true;
                break;
            }
        }
        if (!found) continue;

        float adcv = md->LoaderAutoDetection(buffer, bufferSize);
        Log::DefaultLog.WriteInfo(100, "Module %s auto-detect confidence %f\n", md->ClassName(), adcv);
        if (adcv > 0.99) { // enough for me
            dataSrcClass = md;
            break;
        }
    }

    if (dataSrcClass == NULL) {
        // brute force auto-detection
        // di = ModuleDescriptionManager::Instance()->GetIterator();
        // while (di.HasNext()) {
        //    ModuleDescription *md = di.Next();
        for (auto md : this->GetModuleDescriptionManager()) {
            if (!md->IsLoaderWithAutoDetection()) continue;
            float adcv = md->LoaderAutoDetection(buffer, bufferSize);
            Log::DefaultLog.WriteInfo(100, "Module %s auto-detect confidence %f\n", md->ClassName(), adcv);
            if (adcv > 0.99) { // enough for me
                dataSrcClass = md;
                break;
            }
        }
    }

    delete[] buffer;

    if (dataSrcClass == NULL) {
        Log::DefaultLog.WriteError(
            _T("Failed to Quickstart \"%s\": No suitable data source class found"), filename.PeekBuffer());
        return;
    }

    static int quickstartCounter = 1;
    vislib::StringA viewName;
    vislib::StringA valname;

    viewName.Format("quickstart-%.3d", quickstartCounter);
    ViewDescription view(viewName);

    view.SetViewModuleID("");
    view.AddModule(dataSrcClass, "data");
    valname.Format("data::%s", dataSrcClass->LoaderFilenameSlotName());
    view.AddParamValue(valname, filename);

    valname.Format("quickstartrenderer-%s", dataSrcClass->ClassName());
    if (this->config.IsConfigValueSet(valname)) {
        vislib::StringA rndName(this->config.ConfigValue(valname));
        Log::DefaultLog.WriteInfo(100, "Renderer \"%s\" configured for quickstart", rndName.PeekBuffer());

        // use this renderer
        bool found = false;
        // di = ModuleDescriptionManager::Instance()->GetIterator();
        // while (di.HasNext()) {
        //    ModuleDescription *md = di.Next();
        for (auto md : this->GetModuleDescriptionManager()) {
            if (!rndName.Equals(md->ClassName())) continue;
            view.AddModule(md, "renderer");
            found = true;
            break;
        }

        if (!found) {
            Log::DefaultLog.WriteWarn("Unable to find configured renderer for quickstart. Trying auto-search");
        } else {
            Log::DefaultLog.WriteInfo(50, "Starting module instantiation tests");

            if (!this->quickConnectUp(view, "data", "renderer")) {
                Log::DefaultLog.WriteWarn(
                    "Unable to connect data source with configured renderer for quickstart. Trying auto-search");
                view.ClearModules();
                view.ClearCalls();
                view.ClearParamValues();
                view.SetViewModuleID("");
                view.AddModule(dataSrcClass, "data");
                valname.Format("data::%s", dataSrcClass->LoaderFilenameSlotName());
                view.AddParamValue(valname, filename);

            } else if (!this->quickConnectUp(view, "renderer", NULL)) {
                Log::DefaultLog.WriteWarn("Unable to connect renderer with view for quickstart. Trying auto-search");
                view.ClearModules();
                view.ClearCalls();
                view.ClearParamValues();
                view.SetViewModuleID("");
                view.AddModule(dataSrcClass, "data");
                valname.Format("data::%s", dataSrcClass->LoaderFilenameSlotName());
                view.AddParamValue(valname, filename);

            } else {
                ASSERT(!view.ViewModuleID().IsEmpty());
            }
        }
    } else {
        Log::DefaultLog.WriteInfo(50, "Starting module instantiation tests");
    }

    if (view.ViewModuleID().IsEmpty()) {
        if (!this->quickConnectUp(view, "data", NULL)) {
            Log::DefaultLog.WriteError(
                _T("Failed to Quickstart \"%s\": Cannot auto-connect data source to view"), filename.PeekBuffer());
            return;
        }
        ASSERT(!view.ViewModuleID().IsEmpty());
    }

    std::shared_ptr<ViewDescription> newview(new ViewDescription(view.ClassName()));
    Log::DefaultLog.WriteInfo(
        10, "Quickstart module graph with %u modules and %u calls defined:", view.ModuleCount(), view.CallCount());
    for (unsigned int i = 0; i < view.ModuleCount(); i++) {
        Log::DefaultLog.WriteInfo(25, "Module \"%s\" of class \"%s\"\n", view.Module(i).First().PeekBuffer(),
            view.Module(i).Second()->ClassName());
        newview->AddModule(view.Module(i).Second(), view.Module(i).First());
    }
    for (unsigned int i = 0; i < view.CallCount(); i++) {
        Log::DefaultLog.WriteInfo(25, "Call from \"%s\" to \"%s\" of class \"%s\"\n", view.Call(i).From().PeekBuffer(),
            view.Call(i).To().PeekBuffer(), view.Call(i).Description()->ClassName());
        newview->AddCall(view.Call(i).Description(), view.Call(i).From(), view.Call(i).To());
    }
    for (unsigned int i = 0; i < view.ParamValueCount(); i++) {
        newview->AddParamValue(view.ParamValue(i).First(), view.ParamValue(i).Second());
    }
    newview->SetViewModuleID(view.ViewModuleID());
    this->builtinViewDescs.Register(newview);

    viewName.Format("q%d", quickstartCounter);
    this->RequestViewInstantiation(newview.get(), viewName);
    quickstartCounter++;
    Log::DefaultLog.WriteInfo("Quickstart view instantiation request posted");
}


/*
 * megamol::core::CoreInstance::QuickstartRegistry
 */
void megamol::core::CoreInstance::QuickstartRegistry(const vislib::TString& frontend, const vislib::TString& feparams,
    const vislib::TString& filetype, bool unreg, bool overwrite) {
    using vislib::sys::Log;
    vislib::TString filetypename;

    if (filetype.IsEmpty()) {
        Log::DefaultLog.WriteError("Empty file type is illegal\n");
        return;
    }
    if (filetype.Equals(_T("*"))) {
        // all file types
        // ModuleDescriptionManager::DescriptionIterator di
        //    = ModuleDescriptionManager::Instance()->GetIterator();
        // while (di.HasNext()) {
        //    const ModuleDescription *md = di.Next();
        for (auto md : this->GetModuleDescriptionManager()) {
            if (!md->IsVisibleForQuickstart()) continue;
            const char* fnextsstr = md->LoaderAutoDetectionFilenameExtensions();
            const char* fnnamestr = md->LoaderFileTypeName();
            if (fnextsstr == NULL) continue;
            vislib::Array<vislib::TString> fnexts = vislib::TStringTokeniser::Split(A2T(fnextsstr), _T(";"), true);
            for (SIZE_T i = 0; i < fnexts.Count(); i++) {
                filetypename.Format(_T("%s File"),
                    ((fnnamestr == NULL) ? fnexts[i].Substring(1) : vislib::TString(fnnamestr)).PeekBuffer());
                if (unreg) {
                    this->unregisterQuickstart(frontend, feparams, fnexts[i], filetypename, !overwrite);
                } else {
                    this->registerQuickstart(frontend, feparams, fnexts[i], filetypename, !overwrite);
                }
            }
        }

        return;
    }

    vislib::TString fnext(filetype);
    if (fnext[0] != _T('.')) fnext.Prepend(_T("."));
    // ModuleDescriptionManager::DescriptionIterator di
    //    = ModuleDescriptionManager::Instance()->GetIterator();
    // while (di.HasNext()) {
    //    const ModuleDescription *md = di.Next();
    for (auto md : this->GetModuleDescriptionManager()) {
        if (!md->IsVisibleForQuickstart()) continue;
        const char* fnextsstr = md->LoaderAutoDetectionFilenameExtensions();
        if (fnextsstr == NULL) continue;
        vislib::Array<vislib::TString> fnexts = vislib::TStringTokeniser::Split(A2T(fnextsstr), _T(";"), true);
        for (SIZE_T i = 0; i < fnexts.Count(); i++) {
            if (fnexts[i].Equals(fnext, false)) {
                const char* fnnamestr = md->LoaderFileTypeName();
                filetypename.Format(_T("%s File"),
                    ((fnnamestr == NULL) ? fnext.Substring(1) : vislib::TString(fnnamestr)).PeekBuffer());
                if (unreg) {
                    this->unregisterQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
                } else {
                    this->registerQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
                }
                return;
            }
        }
    }

    Log::DefaultLog.WriteWarn(
        _T("Quickstart %sregistration for unknown type %s"), ((unreg) ? _T("un") : _T("")), filetype.PeekBuffer());
    filetypename.Format(_T("%s File"), filetype.PeekBuffer());
    if (unreg) {
        this->unregisterQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
    } else {
        this->registerQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
    }
}

/*
 * megamol::core::CoreInstance::addProject::WriteStateToXML
 */
bool megamol::core::CoreInstance::WriteStateToXML(const char* outFilename) {
    // using namespace vislib::sys;

    vislib::sys::FastFile outfile;
    if (!outfile.Open(outFilename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ,
            vislib::sys::File::CREATE_OVERWRITE)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to create state file file.");
        return false;
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO, "State has been written to '%s'", outFilename);
    }

    // Write root tag and 'header'
    WriteLineToFile(outfile, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    WriteLineToFile(outfile, "<!--\n");
    WriteLineToFile(outfile, "This file has been auto-generated by MegaMol.\n");
    WriteLineToFile(outfile, "-->\n");
    WriteLineToFile(outfile, "<MegaMol type=\"project\" version=\"1.0\">\n");

    // Loop through active view instances
    // collect data
    vislib::Stack<AbstractNamedObject::ptr_type> stack;
    stack.Push(this->namespaceRoot);
    // auto itervd = this->projViewDescs.GetIterator();
    // auto iterjd = this->projJobDescs.GetIterator();

    int nViews = 1;
    int nJobs = 1;
    vislib::StringA closeModuleTags = "";
    vislib::StringA closeViewJobTags = "";
    vislib::StringA callDesc = "";
    while (!stack.IsEmpty()) {
        AbstractNamedObject::ptr_type ano = stack.Pop();
        ASSERT(ano);
        AbstractNamedObjectContainer::ptr_type anoc = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(ano);
        Module* mod = dynamic_cast<Module*>(ano.get());
        // CalleeSlot *callee = dynamic_cast<CalleeSlot *>(ano.get());
        CallerSlot* caller = dynamic_cast<CallerSlot*>(ano.get());
        param::ParamSlot* param = dynamic_cast<param::ParamSlot*>(ano.get());

        megamol::core::ViewInstance* vi = dynamic_cast<megamol::core::ViewInstance*>(ano.get());
        megamol::core::JobInstance* ji = dynamic_cast<megamol::core::JobInstance*>(ano.get());

        if (anoc != NULL) {
            //            printf("NAMED OBJECT: %s\n", anoc->FullName().PeekBuffer());
            AbstractNamedObjectContainer::child_list_type::iterator i, e;
            e = anoc->ChildList_End();
            for (i = anoc->ChildList_Begin(); i != e; ++i) {
                stack.Push(*i);
            }
        }

        if (vi != NULL) {
            // Write stacked closing tags
            WriteLineToFile(outfile, closeModuleTags.PeekBuffer());
            closeModuleTags.Clear();
            // Write call descriptions if necessary
            WriteLineToFile(outfile, callDesc.PeekBuffer());
            callDesc.Clear();
            // Close previous views/jobs
            WriteLineToFile(outfile, closeViewJobTags.PeekBuffer());
            closeViewJobTags.Clear();

            //            printf("VIEW INSTANCE %s %s\n", vi->FullName().PeekBuffer(), vi->Name().PeekBuffer());
            vislib::StringA classname = "view";
            std::string s = std::to_string(nViews);
            classname.Append(s.c_str());
            nViews++;

            // Write opening tag
            WriteLineToFile(outfile, "  <!--  ");
            WriteLineToFile(outfile, classname.PeekBuffer());
            WriteLineToFile(outfile, "  -->\n");
            WriteLineToFile(outfile, "  <view name=\"");
            WriteLineToFile(outfile, classname.PeekBuffer());
            WriteLineToFile(outfile, "\" viewmod=\"");
            WriteLineToFile(outfile, vi->View()->FullName().PeekBuffer());
            WriteLineToFile(outfile, "\">\n");

            // Stack closing tags closing tag
            closeViewJobTags.Append("  </view>\n");
        }

        if (ji != NULL) {
            // Write stacked closing tags
            WriteLineToFile(outfile, closeModuleTags.PeekBuffer());
            closeModuleTags.Clear();
            // Write call descriptions if necessary
            WriteLineToFile(outfile, callDesc.PeekBuffer());
            callDesc.Clear();
            // Close previous views/jobs
            WriteLineToFile(outfile, closeViewJobTags.PeekBuffer());
            closeViewJobTags.Clear();

            vislib::StringA classname = "job";
            std::string s = std::to_string(nJobs);
            classname.Append(s.c_str());
            nJobs++;
            // TODO this is a little hacky
            Module* mod = dynamic_cast<Module*>(ji->Job());
            if (mod != NULL) {
                // Write opening tag
                WriteLineToFile(outfile, "  <!--  ");
                WriteLineToFile(outfile, classname.PeekBuffer());
                WriteLineToFile(outfile, "  -->\n");
                WriteLineToFile(outfile, "  <job name=\"");
                WriteLineToFile(outfile, classname.PeekBuffer());
                WriteLineToFile(outfile, "\" jobmod=\"");
                WriteLineToFile(outfile, mod->FullName().PeekBuffer());
                WriteLineToFile(outfile, "\">\n");
                // Stack closing tags closing tag
                closeViewJobTags.Append("  </job>\n");
            }
        }

        if (mod != NULL) {
            // Write stacked closing tags
            WriteLineToFile(outfile, closeModuleTags.PeekBuffer());
            closeModuleTags.Clear();

            factories::ModuleDescription::ptr d;
            // ModuleDescriptionManager::DescriptionIterator i =
            //        ModuleDescriptionManager::Instance()->GetIterator();
            // while (i.HasNext()) {
            //    ModuleDescription *id = i.Next();
            for (auto id : this->GetModuleDescriptionManager()) {
                if (id->IsDescribing(mod)) {
                    d = id;
                    break;
                }
            }
            ASSERT(d != NULL);

            // Found a module
            //            printf("    MODULE %s %s\n", d->ClassName(), mod->FullName().PeekBuffer());
            ////            modClasses.Append(d->ClassName());
            ////            modNames.Append(mod->FullName());

            // Write opening tag
            WriteLineToFile(outfile, "    <module class=\"");
            WriteLineToFile(outfile, d->ClassName());
            WriteLineToFile(outfile, "\" name=\"");
            WriteLineToFile(outfile, mod->FullName().PeekBuffer());
            WriteLineToFile(outfile, "\">\n");

            // Stack closing tag
            closeModuleTags.Append("    </module>\n");
        }

        if (caller != NULL) {
            //            // Write stacked closing tags
            //            WriteLineToFile(outfile, closeModuleTags.PeekBuffer());
            //            closeModuleTags.Clear();


            Call* c = caller->CallAs<Call>();
            if (c == NULL) continue;
            factories::CallDescription::ptr d;
            // CallDescriptionManager::DescriptionIterator i =
            //        CallDescriptionManager::Instance()->GetIterator();
            // while (i.HasNext()) {
            //    CallDescription *id = i.Next();
            for (auto id : this->GetCallDescriptionManager()) {
                if (id->IsDescribing(c)) {
                    d = id;
                    break;
                }
            }
            ASSERT(d != NULL);
            ASSERT(c->PeekCalleeSlot() != NULL);
            ASSERT(c->PeekCallerSlot() != NULL);

            //            printf("    CALL %s FROM %s TO %s\n", d->ClassName(),
            //                    c->PeekCallerSlot()->FullName().PeekBuffer(),
            //                    c->PeekCalleeSlot()->FullName().PeekBuffer());
            //
            //            printf("    CALL %s FROM %s TO %s\n", d->ClassName(),
            //                    c->PeekCallerSlot()->Name().PeekBuffer(),
            //                    c->PeekCalleeSlot()->Name().PeekBuffer());

            callDesc.Append("    <call class=\"");
            callDesc.Append(d->ClassName());
            callDesc.Append("\" from=\"");
            callDesc.Append(c->PeekCallerSlot()->FullName().PeekBuffer());
            callDesc.Append("\" to=\"");
            callDesc.Append(c->PeekCalleeSlot()->FullName().PeekBuffer());
            callDesc.Append("\" />\n");
        }

        if (param != NULL) {
            if (param->Parameter().IsNull()) continue;
            //            printf("        PARAM %s = %s\n", param->FullName().PeekBuffer(),
            //                    param->Parameter()->ValueString().PeekBuffer());

            // Write button parameters commented out, since we do not want
            // to trigger them by default
            if (param->Param<param::ButtonParam>() != NULL) {
                WriteLineToFile(outfile, "      <!--<param name=\"");
                WriteLineToFile(outfile, param->FullName().PeekBuffer());
                WriteLineToFile(outfile, "\" value=\"");
#ifdef WIN32
                vislib::sys::WriteLineToFile<char>(outfile, param->Parameter()->ValueString().PeekBuffer());
#else
                // TODO This does not work in windows
                // Here we would need W2A(param->Parameter()->ValueString().PeekBuffer()),
                // however, that does not compile under linux
                vislib::sys::WriteLineToFile<char>(outfile, param->Parameter()->ValueString().PeekBuffer());
#endif
                WriteLineToFile(outfile, "\" />-->\n");
            } else {
                WriteLineToFile(outfile, "      <param name=\"");
                WriteLineToFile(outfile, param->FullName().PeekBuffer());
                WriteLineToFile(outfile, "\" value=\"");
#ifdef WIN32
                vislib::sys::WriteLineToFile<char>(outfile, param->Parameter()->ValueString().PeekBuffer());
#else
                // TODO This does not work in windows
                // Here we would need W2A(param->Parameter()->ValueString().PeekBuffer()),
                // however, that does not compile under linux
                vislib::sys::WriteLineToFile<char>(outfile, param->Parameter()->ValueString().PeekBuffer());
#endif
                WriteLineToFile(outfile, "\" />\n");
            }
        }
    }

    // Write stacked closing tags
    WriteLineToFile(outfile, closeModuleTags.PeekBuffer());
    // Write call descriptions if necessary
    WriteLineToFile(outfile, callDesc.PeekBuffer());
    // Close previous views/jobs
    WriteLineToFile(outfile, closeViewJobTags.PeekBuffer());

    // Close root tag
    WriteLineToFile(outfile, "</MegaMol>");

    outfile.Close();

    return true;
}


/*
 * megamol::core::CoreInstance::InstallService
 */
unsigned int megamol::core::CoreInstance::InstallServiceObject(
    megamol::core::AbstractService* service, ServiceDeletor deletor) {
    assert(services != nullptr);
    return services->InstallServiceObject(service, deletor);
}


/*
 * megamol::core::CoreInstance::GetInstalledService
 */
megamol::core::AbstractService* megamol::core::CoreInstance::GetInstalledService(unsigned int id) {
    assert(services != nullptr);
    return services->GetInstalledService(id);
}


/*
 * megamol::core::CoreInstance::addProject
 */
void megamol::core::CoreInstance::addProject(megamol::core::utility::xml::XmlReader& reader) {
    using vislib::sys::Log;
    utility::ProjectParser parser(this);
    if (parser.Parse(reader)) {
        // success, add project elements
        std::shared_ptr<ViewDescription> vd;
        while (true) {
            vd = parser.PopViewDescription();
            if (vd != NULL) {
                this->projViewDescs.Register(vd);
            } else
                break;
        }
        std::shared_ptr<JobDescription> jd;
        while (true) {
            jd = parser.PopJobDescription();
            if (jd != NULL) {
                this->projJobDescs.Register(jd);
            } else
                break;
        }
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse project file:");
        vislib::SingleLinkedList<vislib::StringA>::Iterator msgs = parser.Messages();
        while (msgs.HasNext()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s\n", msgs.Next().PeekBuffer());
        }
    }
}


#if defined(DEBUG) || defined(_DEBUG)

/*
 * debugDumpSlots
 */
void debugDumpSlots(megamol::core::AbstractNamedObjectContainer* c) {
    using megamol::core::AbstractNamedObject;
    using megamol::core::AbstractNamedObjectContainer;
    using megamol::core::CalleeSlot;
    using megamol::core::CallerSlot;
    using megamol::core::param::ParamSlot;
    using vislib::sys::Log;

    auto e = c->ChildList_End();
    for (auto b = c->ChildList_Begin(); b != e; ++b) {
        AbstractNamedObject::ptr_type ano = *b;
        AbstractNamedObjectContainer::ptr_type anoc = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(ano);
        CalleeSlot* callee = dynamic_cast<CalleeSlot*>(ano.get());
        CallerSlot* caller = dynamic_cast<CallerSlot*>(ano.get());
        ParamSlot* param = dynamic_cast<ParamSlot*>(ano.get());

        if (callee != NULL) {
            Log::DefaultLog.WriteInfo(150, "Callee Slot: %s", callee->FullName().PeekBuffer());
        } else if (caller != NULL) {
            Log::DefaultLog.WriteInfo(150, "Caller Slot: %s", caller->FullName().PeekBuffer());
        } else if (param != NULL) {
            Log::DefaultLog.WriteInfo(150, "Param Slot: %s", param->FullName().PeekBuffer());
        } else if (anoc != NULL) {
            debugDumpSlots(anoc.get());
        }
    }
}

#endif /* DEBUG || _DEBUG */


/*
 * megamol::core::CoreInstance::instantiateModule
 */
megamol::core::Module::ptr_type megamol::core::CoreInstance::instantiateModule(
    const vislib::StringA path, factories::ModuleDescription::ptr desc) {
    using vislib::sys::Log;

    ASSERT(path.StartsWith("::"));
    ASSERT(desc != NULL);

    if (!desc->IsAvailable()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to make module \"%s\" (%s): Module type is installed but not available.", desc->ClassName(),
            path.PeekBuffer());
        return Module::ptr_type(nullptr);
    }

    vislib::Array<vislib::StringA> dirs = vislib::StringTokeniserA::Split(path, "::", true);
    vislib::StringA modName = dirs.Last();
    dirs.RemoveLast();

    ModuleNamespace::ptr_type cns =
        ModuleNamespace::dynamic_pointer_cast(this->namespaceRoot->FindNamespace(dirs, true));
    if (!cns) return Module::ptr_type(nullptr);

    AbstractNamedObject::ptr_type ano = cns->FindChild(modName);
    if (ano) {
        Module::ptr_type tstMod = std::dynamic_pointer_cast<Module>(ano);
        if ((tstMod != NULL) && (desc->IsDescribing(tstMod.get()))) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_WARN, "Unable to make module \"%s\": module already present", path.PeekBuffer());
            return tstMod;
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to make module \"%s\" (%s): name conflict with other namespace object.", desc->ClassName(),
            path.PeekBuffer());
        return Module::ptr_type(nullptr);
    }

    Module::ptr_type mod = Module::ptr_type(desc->CreateModule(modName));
    if (!mod) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to construct module \"%s\" (%s)", desc->ClassName(), path.PeekBuffer());

    } else {
        std::shared_ptr<RootModuleNamespace> tmpRoot = std::make_shared<RootModuleNamespace>();
        tmpRoot->SetCoreInstance(*this);
        tmpRoot->AddChild(mod);

        if (!mod->Create()) {
            tmpRoot->RemoveChild(mod);
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Unable to create module \"%s\" (%s)", desc->ClassName(), path.PeekBuffer());
            mod.reset();

        } else {
            tmpRoot->RemoveChild(mod);
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_INFO + 350, "Created module \"%s\" (%s)", desc->ClassName(), path.PeekBuffer());
            cns->AddChild(mod);
#if defined(DEBUG) || defined(_DEBUG)
            debugDumpSlots(mod.get());
#endif /* DEBUG || _DEBUG */
        }
    }

    return mod;
}


/*
 * megamol::core::CoreInstance::InstantiateCall
 */
megamol::core::Call* megamol::core::CoreInstance::InstantiateCall(
    const vislib::StringA fromPath, const vislib::StringA toPath, megamol::core::factories::CallDescription::ptr desc) {
    using vislib::sys::Log;

    ASSERT(fromPath.StartsWith("::"));
    ASSERT(toPath.StartsWith("::"));
    ASSERT(desc != NULL);

    vislib::Array<vislib::StringA> fromDirs = vislib::StringTokeniserA::Split(fromPath, "::", true);
    vislib::StringA fromSlotName = fromDirs.Last();
    fromDirs.RemoveLast();
    vislib::StringA fromModName = fromDirs.Last();
    fromDirs.RemoveLast();

    vislib::Array<vislib::StringA> toDirs = vislib::StringTokeniserA::Split(toPath, "::", true);
    vislib::StringA toSlotName = toDirs.Last();
    toDirs.RemoveLast();
    vislib::StringA toModName = toDirs.Last();
    toDirs.RemoveLast();

    ModuleNamespace::ptr_type fromNS =
        ModuleNamespace::dynamic_pointer_cast(this->namespaceRoot->FindNamespace(fromDirs, false));
    if (!fromNS) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to instantiate call: can not find source namespace \"%s\"",
            fromPath.PeekBuffer());
        return NULL;
    }

    ModuleNamespace::ptr_type toNS =
        ModuleNamespace::dynamic_pointer_cast(this->namespaceRoot->FindNamespace(toDirs, false));
    if (!toNS) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: can not find target namespace \"%s\"", toPath.PeekBuffer());
        return NULL;
    }

    Module::ptr_type fromMod = std::dynamic_pointer_cast<Module>(fromNS->FindChild(fromModName));
    if (!fromMod) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: can not find source module \"%s\"", fromPath.PeekBuffer());
        return NULL;
    }

    Module::ptr_type toMod = std::dynamic_pointer_cast<Module>(toNS->FindChild(toModName));
    if (!toMod) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: can not find target module \"%s\"", toPath.PeekBuffer());
        return NULL;
    }

    CallerSlot* fromSlot = dynamic_cast<CallerSlot*>(fromMod->FindSlot(fromSlotName));
    if (fromSlot == NULL) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: can not find source slot \"%s\"", fromPath.PeekBuffer());
        return NULL;
    }

    CalleeSlot* toSlot = dynamic_cast<CalleeSlot*>(toMod->FindSlot(toSlotName));
    if (toSlot == NULL) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: can not find target slot \"%s\"", toPath.PeekBuffer());
        return NULL;
    }

    if (!fromSlot->IsCallCompatible(desc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: source slot \"%s\" not compatible with call \"%s\"", fromPath.PeekBuffer(),
            desc->ClassName());
        return NULL;
    }

    if (!toSlot->IsCallCompatible(desc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: target slot \"%s\" not compatible with call \"%s\"", toPath.PeekBuffer(),
            desc->ClassName());
        return NULL;
    }

    if ((fromSlot->GetStatus() == AbstractSlot::STATUS_CONNECTED) &&
        (toSlot->GetStatus() == AbstractSlot::STATUS_CONNECTED)) {
        Call* tstCall = fromSlot->IsConnectedTo(toSlot);
        if (tstCall != NULL) {
            if (desc->IsDescribing(tstCall)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Unable to instantiate call \"%s\"->\"%s\": call already exists", fromPath.PeekBuffer(),
                    toPath.PeekBuffer());
                return tstCall;
            }
        }
    }

    if (fromSlot->GetStatus() != AbstractSlot::STATUS_ENABLED) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: status of source slot \"%s\" is invalid.", fromPath.PeekBuffer());
        return NULL;
    }

    Call* call = desc->CreateCall();
    if (call == NULL) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to instantiate call: failed to create call \"%s\"", desc->ClassName());
        return NULL;
    }

    if (!toSlot->ConnectCall(call)) {
        delete call;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: failed to connect call \"%s\" to target slot \"%s\"", desc->ClassName(),
            toPath.PeekBuffer());
        return NULL;
    }

    if (!fromSlot->ConnectCall(call)) {
        delete call; // Disconnects call as sfx
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: failed to connect call \"%s\" to source slot \"%s\"", desc->ClassName(),
            fromPath.PeekBuffer());
        return NULL;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350, "Call \"%s\" instantiated from \"%s\" to \"%s\"", desc->ClassName(),
        fromPath.PeekBuffer(), toPath.PeekBuffer());

    return call;
}


/*
 * megamol::core::CoreInstance::enumParameters
 */
void megamol::core::CoreInstance::enumParameters(megamol::core::ModuleNamespace::const_ptr_type path,
    std::function<void(const Module&, param::ParamSlot&)> cb) const {
    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    // TODO use EnumModulesNoLock?!

    AbstractNamedObjectContainer::child_list_type::const_iterator i, e;
    e = path->ChildList_End();
    for (i = path->ChildList_Begin(); i != e; ++i) {
        AbstractNamedObject::const_ptr_type child = *i;
        Module::const_ptr_type mod = Module::dynamic_pointer_cast(child);
        ModuleNamespace::const_ptr_type ns = ModuleNamespace::dynamic_pointer_cast(child);

        if (mod) {
            AbstractNamedObjectContainer::child_list_type::const_iterator si, se;
            se = mod->ChildList_End();
            for (si = mod->ChildList_Begin(); si != se; ++si) {
                param::ParamSlot* slot = dynamic_cast<param::ParamSlot*>((*si).get());
                if (slot) {
                    cb(*mod, *slot);
                }
            }

        } else if (ns) {
            this->enumParameters(ns, cb);
        }
    }
}


/*
 * megamol::core::CoreInstance::findParameterName
 */
vislib::StringA megamol::core::CoreInstance::findParameterName(megamol::core::ModuleNamespace::const_ptr_type path,
    const vislib::SmartPtr<megamol::core::param::AbstractParam>& param) const {

    vislib::sys::AutoLock lock(this->namespaceRoot->ModuleGraphLock());

    AbstractNamedObjectContainer::child_list_type::const_iterator i, e;
    e = path->ChildList_End();
    for (i = path->ChildList_Begin(); i != e; ++i) {
        AbstractNamedObject::const_ptr_type child = *i;
        Module::const_ptr_type mod = Module::dynamic_pointer_cast(child);
        ModuleNamespace::const_ptr_type ns = ModuleNamespace::dynamic_pointer_cast(child);

        if (mod) {
            AbstractNamedObjectContainer::child_list_type::const_iterator si, se;
            se = mod->ChildList_End();
            for (si = mod->ChildList_Begin(); si != se; ++si) {
                const param::ParamSlot* slot = dynamic_cast<const param::ParamSlot*>((*si).get());
                if ((slot != NULL) && (slot->Parameter() == param)) {
                    vislib::StringA name(mod->FullName());
                    name.Append("::");
                    name.Append(slot->Name());
                    return name;
                }
            }

        } else if (ns) {
            vislib::StringA n = this->findParameterName(ns, param);
            if (!n.IsEmpty()) {
                return n;
            }
        }
    }
    return "";
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
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO + 50, "View instance %s terminating ...", vi->Name().PeekBuffer());
        vi->Terminate();
    }
    if (ji != NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO + 50, "Job instance %s terminating ...", ji->Name().PeekBuffer());
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
 * megamol::core::CoreInstance::applyConfigParams
 */
void megamol::core::CoreInstance::applyConfigParams(
    const vislib::StringA& root, const InstanceDescription* id, const ParamValueSetRequest* params) {

    for (unsigned int i = 0; i < id->ParamValueCount(); i++) {
        const InstanceDescription::ParamValueRequest& pvr = id->ParamValue(i);
        vislib::StringA nameA(pvr.First());
        if (!nameA.StartsWith("::")) {
            nameA.Prepend("::");
            nameA.Prepend(root);
            if (!nameA.StartsWith("::")) {
                nameA.Prepend("::");
            }
        }

        vislib::SmartPtr<param::AbstractParam> p = this->FindParameter(nameA, true);
        if (!p.IsNull()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Initializing parameter \"%s\" to \"%s\"", nameA.PeekBuffer(),
                vislib::StringA(pvr.Second()).PeekBuffer());
            p->ParseValue(pvr.Second());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "Unable to set parameter \"%s\" to \"%s\": parameter not found", nameA.PeekBuffer(),
                vislib::StringA(pvr.Second()).PeekBuffer());
        }
    }

    for (unsigned int i = 0; i < params->ParamValueCount(); i++) {
        const ParamValueSetRequest::ParamValueRequest& pvr = params->ParamValue(i);
        vislib::StringA nameA(pvr.First());
        if (!nameA.StartsWith("::")) {
            nameA.Prepend("::");
            nameA.Prepend(root);
            if (!nameA.StartsWith("::")) {
                nameA.Prepend("::");
            }
        }

        vislib::SmartPtr<param::AbstractParam> p = this->FindParameter(nameA, true);
        if (!p.IsNull()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Setting parameter \"%s\" to \"%s\"",
                nameA.PeekBuffer(), vislib::StringA(pvr.Second()).PeekBuffer());
            p->ParseValue(pvr.Second());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "Unable to set parameter \"%s\" to \"%s\": parameter not found", nameA.PeekBuffer(),
                vislib::StringA(pvr.Second()).PeekBuffer());
        }
    }
}


/*
 * megamol::core::CoreInstance::loadPlugin
 */
void megamol::core::CoreInstance::loadPlugin(const vislib::TString& filename) {

    // select log level for plugin loading errors
    unsigned int loadFailedLevel = vislib::sys::Log::LEVEL_ERROR;
    if (this->config.IsConfigValueSet("PluginLoadFailMsg")) {
        try {
            const vislib::StringW& v = this->config.ConfigValue("PluginLoadFailMsg");
            if (v.Equals(L"error", false) || v.Equals(L"err", false) || v.Equals(L"e", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_ERROR;
            } else if (v.Equals(L"warning", false) || v.Equals(L"warn", false) || v.Equals(L"w", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_WARN;
            } else if (v.Equals(L"information", false) || v.Equals(L"info", false) || v.Equals(L"i", false) ||
                       v.Equals(L"message", false) || v.Equals(L"msg", false) || v.Equals(L"m", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_INFO;
            } else {
                loadFailedLevel = vislib::CharTraitsW::ParseInt(v.PeekBuffer());
            }
        } catch (...) {
        }
    }

    try {

        utility::plugins::PluginManager::collection_type new_plugins =
            this->plugins->LoadPlugin(filename.PeekBuffer(), *this);

        for (auto new_plugin : new_plugins) {
            this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Plugin \"%s\" (%s) loaded: %d Modules, %d Calls registered\n", new_plugin->GetAssemblyName().c_str(),
                vislib::StringA(filename).PeekBuffer(), new_plugin->GetModuleDescriptionManager().Count(),
                new_plugin->GetCallDescriptionManager().Count());

            for (auto md : new_plugin->GetModuleDescriptionManager()) {
                try {
                    this->all_module_descriptions.Register(md);
                } catch (const vislib::AlreadyExistsException&) {
                    this->log.WriteError("Failed to load module description \"%s\": Naming conflict", md->ClassName());
                }
            }
            for (auto cd : new_plugin->GetCallDescriptionManager()) {
                try {
                    this->all_call_descriptions.Register(cd);
                } catch (const vislib::AlreadyExistsException&) {
                    this->log.WriteError("Failed to load call description \"%s\": Naming conflict", cd->ClassName());
                }
            }
        }

    } catch (const vislib::Exception& vex) {
        this->log.WriteMsg(loadFailedLevel, "Unable to load Plugin \"%s\": %s (%s, &d)",
            vislib::StringA(filename).PeekBuffer(), vex.GetMsgA(), vex.GetFile(), vex.GetLine());
    } catch (...) {
        this->log.WriteMsg(
            loadFailedLevel, "Unable to load Plugin \"%s\": unknown exception", vislib::StringA(filename).PeekBuffer());
    }
}


/*
 * megamol::core::CoreInstance::mapCompare
 */
bool megamol::core::CoreInstance::mapCompare(ParamHashMap_t& one, ParamHashMap_t& other) {
    if (one.size() != other.size()) return false;

    for (auto& entry : one) {
        auto entry2 = other.find(entry.first);
        if (entry2 == other.end() || entry.second != entry2->second) return false;
    }

    return true;
}


/*
 * megamol::core::CoreInstance::quickConnectUp
 */
bool megamol::core::CoreInstance::quickConnectUp(
    megamol::core::ViewDescription& view, const char* from, const char* to) {
    using vislib::sys::Log;

    vislib::SingleLinkedList<vislib::Array<quickStepInfo>> fifo;
    vislib::Array<quickStepInfo> connInfo;
    vislib::StringA toClass;
    quickStepInfo qsi;

    qsi.call = NULL;
    qsi.nextSlot = NULL;
    qsi.prevSlot = NULL;
    qsi.nextMod = NULL;
    if (to != NULL) {
        for (unsigned int i = 0; i < view.ModuleCount(); i++) {
            if (view.Module(i).First().Equals(to, false)) {
                qsi.nextMod = view.Module(i).Second();
                break;
            }
        }
        if (qsi.nextMod == NULL) {
            Log::DefaultLog.WriteError("Internal search name error #%d\n", __LINE__);
            return false;
        }
        toClass = qsi.nextMod->ClassName();
        qsi.nextMod = NULL;
    }
    for (unsigned int i = 0; i < view.ModuleCount(); i++) {
        if (view.Module(i).First().Equals(from)) {
            qsi.nextMod = view.Module(i).Second();
            break;
        }
    }
    if (qsi.nextMod == NULL) {
        Log::DefaultLog.WriteError("Internal search name error #%d\n", __LINE__);
        return false;
    }
    fifo.Append(vislib::Array<quickStepInfo>(1, qsi));

    while (!fifo.IsEmpty()) {
        vislib::Array<quickStepInfo> list(fifo.First());
        fifo.RemoveFirst();
        const quickStepInfo& lqsi = list.Last();
        ASSERT(lqsi.nextMod != NULL);

        // printf("Test connection from %s\n", lqsi.nextMod->ClassName());
        this->quickConnectUpStepInfo(lqsi.nextMod, connInfo);

        // test for end condition
        if (to == NULL) {
            for (SIZE_T i = 0; i < connInfo.Count(); i++) {
                if (vislib::StringA("View2D").Equals(connInfo[i].nextMod->ClassName(), false) ||
                    vislib::StringA("View3D").Equals(connInfo[i].nextMod->ClassName(), false) ||
                    vislib::StringA("View3D_2").Equals(connInfo[i].nextMod->ClassName(), false)) {

                    vislib::StringA prevModName(from);
                    for (SIZE_T j = 1; j < list.Count(); j++) {
                        vislib::StringA modName;
                        modName.Format("mod%.3d", view.ModuleCount());
                        view.AddModule(list[j].nextMod, modName);
                        view.AddCall(
                            list[j].call, modName + "::" + list[j].nextSlot, prevModName + "::" + list[j].prevSlot);
                        prevModName = modName;
                    }

                    view.AddModule(connInfo[i].nextMod, "view");
                    view.AddCall(connInfo[i].call, vislib::StringA("view::") + connInfo[i].nextSlot,
                        prevModName + "::" + connInfo[i].prevSlot);
                    view.SetViewModuleID("view");
                    return true;
                }
            }

        } else {
            for (SIZE_T i = 0; i < connInfo.Count(); i++) {
                if (vislib::StringA(toClass).Equals(connInfo[i].nextMod->ClassName(), false)) {

                    vislib::StringA prevModName(from);
                    for (SIZE_T j = 1; j < list.Count(); j++) {
                        vislib::StringA modName;
                        modName.Format("mod%.3d", view.ModuleCount());
                        view.AddModule(list[j].nextMod, modName);
                        view.AddCall(
                            list[j].call, modName + "::" + list[j].nextSlot, prevModName + "::" + list[j].prevSlot);
                        prevModName = modName;
                    }

                    view.AddCall(connInfo[i].call, vislib::StringA(to) + "::" + connInfo[i].nextSlot,
                        prevModName + "::" + connInfo[i].prevSlot);
                    return true;
                }
            }
        }

        // continue search
        for (SIZE_T i = 0; i < connInfo.Count(); i++) {
            bool add = true; // no cycles!
            for (SIZE_T j = 0; j < list.Count(); j++) {
                if (vislib::StringA(list[j].nextMod->ClassName()).Equals(connInfo[i].nextMod->ClassName())) {
                    add = false;
                    break;
                }
            }
            if (!add) continue;
            vislib::Array<quickStepInfo> newlist(list);
            newlist.Append(connInfo[i]);
            fifo.Append(newlist);
        }
    }

    // Nothing found
    return false;
}


/*
 * megamol::core::CoreInstance::quickConnectUpStepInfo
 */
void megamol::core::CoreInstance::quickConnectUpStepInfo(megamol::core::factories::ModuleDescription::ptr from,
    vislib::Array<megamol::core::CoreInstance::quickStepInfo>& step) {
    using vislib::sys::Log;
    ASSERT(from != NULL);
    step.Clear();

    Module::ptr_type m(from->CreateModule("quickstarttest"));
    if (!m) {
        Log::DefaultLog.WriteError("Unable to test-instantiate module %s", from->ClassName());
        return;
    }

    vislib::SingleLinkedList<vislib::Pair<factories::CallDescription::ptr, vislib::StringA>> inCalls;
    vislib::Stack<AbstractNamedObjectContainer::ptr_type> stack(m);
    while (!stack.IsEmpty()) {
        AbstractNamedObjectContainer::ptr_type anoc = stack.Pop();
        AbstractNamedObjectContainer::child_list_type::iterator ci, ce;
        ce = anoc->ChildList_End();
        for (ci = anoc->ChildList_Begin(); ci != ce; ++ci) {
            AbstractNamedObject::ptr_type ano = *ci;
            AbstractNamedObjectContainer::ptr_type anoc2 = AbstractNamedObjectContainer::dynamic_pointer_cast(ano);
            if (anoc2) {
                stack.Push(anoc2);
            }
            if (dynamic_cast<CalleeSlot*>(ano.get()) != NULL) {
                CalleeSlot* callee = dynamic_cast<CalleeSlot*>(ano.get());

                // factoirCallDescriptionManager::DescriptionIterator cdi =
                // CallDescriptionManager::Instance()->GetIterator(); while (cdi.HasNext()) {
                //    CallDescription *cd = cdi.Next();
                for (auto cd : this->GetCallDescriptionManager()) {
                    vislib::Pair<factories::CallDescription::ptr, vislib::StringA> cse(cd, callee->Name());
                    if (inCalls.Contains(cse)) continue;
                    if (callee->IsCallCompatible(cd)) {
                        inCalls.Add(cse);
                    }
                }
            }
        }
    }
    m->SetAllCleanupMarks();
    m->PerformCleanup();
    m.reset();
    // now 'inCalls' holds all calls which can connect to 'from' (some slot)

    // ModuleDescriptionManager::DescriptionIterator mdi = ModuleDescriptionManager::Instance()->GetIterator();
    // while (mdi.HasNext()) {
    //    ModuleDescription *md = mdi.Next();
    for (auto md : this->GetModuleDescriptionManager()) {
        if (md == from) continue;
        if (!md->IsVisibleForQuickstart()) continue;
        m = Module::ptr_type(md->CreateModule("quickstarttest"));
        if (!m) continue;

        bool connectable = false;
        stack.Push(m);
        while (!stack.IsEmpty() && !connectable) {
            AbstractNamedObjectContainer::ptr_type anoc = stack.Pop();
            ASSERT(anoc);
            AbstractNamedObjectContainer::child_list_type::iterator ci, ce;
            ce = anoc->ChildList_End();
            for (ci = anoc->ChildList_Begin(); (ci != ce) && !connectable; ++ci) {
                AbstractNamedObject::ptr_type ano = *ci;
                ASSERT(ano);
                AbstractNamedObjectContainer::ptr_type anoc2 = AbstractNamedObjectContainer::dynamic_pointer_cast(ano);
                if (anoc2) {
                    stack.Push(anoc2);
                }
                if (dynamic_cast<CallerSlot*>(ano.get()) != NULL) {
                    CallerSlot* caller = dynamic_cast<CallerSlot*>(ano.get());
                    ASSERT(caller != NULL);

                    vislib::SingleLinkedList<vislib::Pair<factories::CallDescription::ptr, vislib::StringA>>::Iterator
                        cdi = inCalls.GetIterator();
                    while (cdi.HasNext() && !connectable) {
                        const vislib::Pair<factories::CallDescription::ptr, vislib::StringA>& cde = cdi.Next();
                        factories::CallDescription::ptr cd = cde.First();
                        ASSERT(cd != NULL);
                        if (caller->IsCallCompatible(cd)) {
                            connectable = true;
                            stack.Clear();
                            quickStepInfo qsi;
                            qsi.call = cd;
                            qsi.nextMod = md;
                            qsi.nextSlot = caller->Name();
                            qsi.prevSlot = cde.Second();
                            step.Add(qsi);
                        }
                    }
                }
            }
        }
        m->SetAllCleanupMarks();
        m->PerformCleanup();
        VLTRACE(VISLIB_TRCELVL_INFO, "dtoring %s ... ", md->ClassName());
        m.reset();
        VLTRACE(VISLIB_TRCELVL_INFO, "done dtoring.\n");
    }
}


#ifdef _WIN32
extern HMODULE mmCoreModuleHandle;
#endif /* _WIN32 */


/*
 * megamol::core::CoreInstance::registerQuickstart
 */
void megamol::core::CoreInstance::registerQuickstart(const vislib::TString& frontend, const vislib::TString& feparams,
    const vislib::TString& fnext, const vislib::TString& fnname, bool keepothers) {
    using vislib::sys::Log;
#ifdef _WIN32
    using vislib::sys::RegistryKey;
#endif /* _WIN32 */
    ASSERT(!fnext.IsEmpty());
    ASSERT(fnext[0] == _T('.'));
#ifdef _WIN32
    Log::DefaultLog.WriteInfo(
        _T("Registering \"%s\" type (*%s) for quickstart"), fnname.PeekBuffer(), fnext.PeekBuffer());
    try {
        DWORD errcode;
        vislib::TString str;
        RegistryKey crw(RegistryKey::HKeyClassesRoot(), KEY_ALL_ACCESS);
        if (!crw.IsValid()) {
            throw vislib::Exception("Cannot open \"HKEY_CLASSES_ROOT\" for writing", __FILE__, __LINE__);
        }

        RegistryKey extKey;
        errcode = crw.OpenSubKey(extKey, fnext);
        if (errcode != ERROR_SUCCESS) {
            if (errcode == ERROR_FILE_NOT_FOUND) {
                errcode = crw.CreateSubKey(extKey, fnext);
            }
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }

        vislib::TString typeName;
        errcode = extKey.GetValue(_T(""), typeName);
        if (errcode != ERROR_SUCCESS) {
            typeName.Format(_T("MegaMol.%d.%d%s"), MEGAMOL_VERSION_MAJOR, MEGAMOL_VERSION_MINOR, fnext.PeekBuffer());
            errcode = extKey.SetValue(_T(""), typeName);
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
            errcode = extKey.GetValue(_T(""), typeName);
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }
        extKey.Close();

        RegistryKey typeKey;
        errcode = crw.OpenSubKey(typeKey, typeName);
        if (errcode != ERROR_SUCCESS) {
            if (errcode == ERROR_FILE_NOT_FOUND) {
                errcode = crw.CreateSubKey(typeKey, typeName);
            }
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }

        errcode = typeKey.GetValue(_T(""), str);
        if (errcode != ERROR_SUCCESS) {
            typeKey.SetValue(_T(""), fnname);
        }

        RegistryKey defIcon;
        errcode = typeKey.OpenSubKey(defIcon, "DefaultIcon");
        if (errcode != ERROR_SUCCESS) {
            errcode = typeKey.CreateSubKey(defIcon, "DefaultIcon");
        }
        if (errcode == ERROR_SUCCESS) {
            TCHAR fn[64 * 1024];
            DWORD size = 64 * 1024;
            size = ::GetModuleFileName(mmCoreModuleHandle, fn, size);
            str = vislib::TString(fn, size);
            str.Append(_T(",3")); // TODO: howto map: 1001 -> 3
            defIcon.SetValue(_T(""), str);
        }
        defIcon.Close();

        RegistryKey shell;
        errcode = typeKey.OpenSubKey(shell, "shell");
        if (errcode != ERROR_SUCCESS) {
            errcode = typeKey.CreateSubKey(shell, "shell");
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }

        RegistryKey open;
        vislib::TString opencmd;
        opencmd.Format(_T("open.%d.%d.%d.%d"), MEGAMOL_CORE_VERSION);
        errcode = shell.OpenSubKey(open, opencmd);
        if (errcode != ERROR_SUCCESS) {
            errcode = shell.CreateSubKey(open, opencmd);
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }

        str = _T("Open with MegaMol�");
        if (vislib::sys::SystemInformation::SystemWordSize() != vislib::sys::SystemInformation::SelfWordSize()) {
            if (vislib::sys::SystemInformation::SelfWordSize() == 64) {
                str.Append(_T(" x64"));
            } else {
                str.Append(_T(" x86"));
            }
        }
#    if defined(DEBUG) || defined(_DEBUG)
        str.Append(_T(" [Debug]"));
#    endif /* DEBUG || _DEBUG */
        open.SetValue(_T(""), str);

        vislib::TString cmdline = feparams;
        cmdline.Replace(_T("$(FILENAME)"), _T("\"%1\""));
        cmdline.Prepend(_T("\" "));
        cmdline.Prepend(frontend);
        cmdline.Prepend(_T("\""));
        RegistryKey opencommand;
        errcode = open.OpenSubKey(opencommand, "command");
        if (errcode != ERROR_SUCCESS) {
            errcode = open.CreateSubKey(opencommand, "command");
            if (errcode != ERROR_SUCCESS) throw vislib::sys::SystemException(errcode, __FILE__, __LINE__);
        }
        opencommand.SetValue(_T(""), cmdline);
        opencommand.Close();
        open.Close();

        errcode = shell.GetValue(_T(""), str);
        if ((errcode == 2) || !keepothers) {
            shell.SetValue(_T(""), opencmd);
        }

        shell.Close();
        typeKey.Close();
        crw.Close();

    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteError(_T("Cannot register quickstart for %s: %s (%s, %d)"), fnext.PeekBuffer(),
            ex.GetMsg(), vislib::TString(ex.GetFile()).PeekBuffer(), ex.GetLine());
    } catch (...) {
        Log::DefaultLog.WriteError(_T("Cannot register quickstart for %s: Unexpected Exception"), fnext.PeekBuffer());
    }
#else  /* _WIN32 */
    Log::DefaultLog.WriteError(
        _T("Quickstart registration is not supported on this operating system"), fnext.PeekBuffer());
#endif /* _WIN32 */
}


/*
 * megamol::core::CoreInstance::unregisterQuickstart
 */
void megamol::core::CoreInstance::unregisterQuickstart(const vislib::TString& frontend, const vislib::TString& feparams,
    const vislib::TString& fnext, const vislib::TString& fnname, bool keepothers) {
    using vislib::sys::Log;
#ifdef _WIN32
    using vislib::sys::RegistryKey;
#endif /* _WIN32 */
    ASSERT(!fnext.IsEmpty());
    ASSERT(fnext[0] == _T('.'));
#ifdef _WIN32
    Log::DefaultLog.WriteInfo(
        _T("Un-Registering \"%s\" type (*%s) for quickstart"), fnname.PeekBuffer(), fnext.PeekBuffer());
    try {
        DWORD errcode;

        RegistryKey crw(RegistryKey::HKeyClassesRoot(), KEY_ALL_ACCESS);
        if (!crw.IsValid()) {
            throw vislib::Exception("Cannot open \"HKEY_CLASSES_ROOT\" for writing", __FILE__, __LINE__);
        }

        RegistryKey extKey;
        errcode = crw.OpenSubKey(extKey, fnext);
        if (errcode != ERROR_SUCCESS) {
            Log::DefaultLog.WriteWarn(_T("File type %s does not seem to be registered"), fnext.PeekBuffer());
            return;
        }

        vislib::TString typeName;
        errcode = extKey.GetValue(_T(""), typeName);
        if (errcode != ERROR_SUCCESS) {
            Log::DefaultLog.WriteWarn(
                _T("File type %s does not seem to be correctly registered (#0x1)"), fnext.PeekBuffer());
            if (extKey.GetSubKeysA().Count() == 0) {
                extKey.Close();
                crw.DeleteSubKey(fnext); // just delete it already, since it's broken anyway
            }
            return;
        }

        RegistryKey typeKey;
        errcode = crw.OpenSubKey(typeKey, typeName);
        if (errcode != ERROR_SUCCESS) {
            Log::DefaultLog.WriteWarn(
                _T("File type %s does not seem to be correctly registered (#0x2)"), fnext.PeekBuffer());
            if (extKey.GetSubKeysA().Count() == 0) {
                extKey.Close();
                crw.DeleteSubKey(fnext); // just delete it already, since it's broken anyway
            }
            return;
        }
        bool delTypeKeys = false;

        if (keepothers) {
            RegistryKey shell;
            errcode = typeKey.OpenSubKey(shell, "shell");
            if (errcode == ERROR_SUCCESS) {
                vislib::Array<vislib::TString> subkeys =
#    if defined(UNICODE) || defined(_UNICODE)
                    shell.GetSubKeysW();
#    else  /* UNICODE || _UNICODE */
                    shell.GetSubKeysA();
#    endif /* UNICODE || _UNICODE */
                vislib::TString fecmd(frontend);
                fecmd.Prepend(_T("\""));
                fecmd.Append(_T("\" "));

                for (SIZE_T i = 0; i < subkeys.Count(); i++) {
                    RegistryKey subkey;
                    errcode = shell.OpenSubKey(subkey, subkeys[i]);
                    if (errcode != ERROR_SUCCESS) continue;
                    RegistryKey shellCmd;
                    errcode = subkey.OpenSubKey(shellCmd, "command");
                    if (errcode != ERROR_SUCCESS) continue;
                    vislib::TString cmdLine;
                    errcode = shellCmd.GetValue(_T(""), cmdLine);
                    if (errcode != ERROR_SUCCESS) continue;

                    if (!cmdLine.StartsWith(fecmd, false)) continue; // other front-end

                    // this front-end, so delete!

                    shellCmd.Close();
                    subkey.Close();

                    shell.DeleteSubKey(subkeys[i]);
                }

                delTypeKeys = (shell.GetSubKeysA().Count() == 0);

                if (!delTypeKeys) {
                    vislib::TString defCmd;
                    errcode = shell.GetValue(_T(""), defCmd);
                    if ((errcode == ERROR_SUCCESS) && !defCmd.IsEmpty()) {
                        RegistryKey tmpkey;
                        errcode = shell.OpenSubKey(tmpkey, defCmd);
                        if (errcode != ERROR_SUCCESS) {
                            shell.DeleteValue(_T(""));
                        } else {
                            tmpkey.Close();
                        }
                    }
                }

            } else {
                Log::DefaultLog.WriteWarn(
                    _T("File type %s does not seem to be correctly registered (#0x3)"), fnext.PeekBuffer());
                delTypeKeys = true;
            }
        } else {
            delTypeKeys = true;
        }

        typeKey.Close();
        extKey.Close();
        if (delTypeKeys) {
            crw.DeleteSubKey(typeName);
            crw.DeleteSubKey(fnext);
        }

        crw.Close();

    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteError(_T("Cannot unregister quickstart for %s: %s (%s, %d)"), fnext.PeekBuffer(),
            ex.GetMsg(), vislib::TString(ex.GetFile()).PeekBuffer(), ex.GetLine());
    } catch (...) {
        Log::DefaultLog.WriteError(_T("Cannot unregister quickstart for %s: Unexpected Exception"), fnext.PeekBuffer());
    }
#else  /* _WIN32 */
    Log::DefaultLog.WriteWarn(
        _T("Quickstart registration is not supported on this operating system"), fnext.PeekBuffer());
#endif /* _WIN32 */
}


void megamol::core::CoreInstance::updateFlushIdxList(size_t const processedCount, std::vector<size_t>& list) {
    list.erase(list.begin());
    std::transform(list.begin(), list.end(), list.begin(), [processedCount](size_t el) { return el - processedCount; });
}


bool megamol::core::CoreInstance::checkForFlushEvent(size_t const eventIdx, std::vector<size_t>& list) const {
    if (!list.empty()) {
        auto const idx = list.front();

        if (eventIdx > idx) {
            return true;
        }
    }

    return false;
}


void megamol::core::CoreInstance::shortenFlushIdxList(size_t const eventCount, std::vector<size_t>& list) {
    list.erase(
        std::remove_if(list.begin(), list.end(), [eventCount](auto el) { return (eventCount - 1) <= el; }), list.end());
}
