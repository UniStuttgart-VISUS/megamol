/*
 * CoreInstance.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
/* create symbols for the opengl extensions in this object file */
#define GLH_EXT_SINGLE_FILE 1
#if (_MSC_VER > 1000)
#pragma warning(disable: 4996)
#endif /* (_MSC_VER > 1000) */
#include "glh/glh_extensions.h"
#if (_MSC_VER > 1000)
#pragma warning(default: 4996)
#endif /* (_MSC_VER > 1000) */

#include "CoreInstance.h"
#include "ObjectDescriptionManager.h"
#include "AbstractSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Call.h"
#include "CallDescription.h"
#include "CallDescriptionManager.h"
#include "cluster/ClusterController.h"
#include "cluster/ClusterViewMaster.h"
#include "cluster/simple/Server.h"
#include "Module.h"
#include "ModuleDescription.h"
#include "ModuleDescriptionManager.h"
#include "param/ParamSlot.h"
#include "param/StringParam.h"
#include "param/ButtonParam.h"
#include "utility/APIValueUtil.h"
#include "utility/ProjectParser.h"
#include "utility/xml/XmlReader.h"
#include "job/JobThread.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include "vislib/functioncast.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/GUID.h"
#include "vislib/RegistryKey.h"
#include "vislib/Socket.h"
#include "vislib/StackTrace.h"
#include "vislib/StringTokeniser.h"
#include "vislib/SystemInformation.h"
#include "vislib/Trace.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/NetworkInformation.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/vislibversion.h"
#include "productversion.h"
#include "versioninfo.h"
#include "profiler/Manager.h"

#include "vislib/Array.h"
#include "vislib/BufferedFile.h"
#include "vislib/CriticalSection.h"
#include "vislib/Console.h"
#include "vislib/functioncast.h"
#include "vislib/Log.h"
#include "vislib/Map.h"
#include "vislib/memutils.h"
#include "vislib/MultiSz.h"
#include "vislib/Path.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sysfunctions.h"
#include "vislib/ThreadSafeStackTrace.h"
#include "vislib/Trace.h"


/*****************************************************************************/


/*
 * megamol::core::CoreInstance::PreInit::PreInit
 */
megamol::core::CoreInstance::PreInit::PreInit() : cfgFileSet(false), 
        logFileSet(false), logLevelSet(false), logEchoLevelSet(false), 
        cfgFile(), logFile(), logLevel(0), logEchoLevel(0) {
    // atm intentionally empty
}


/*****************************************************************************/


/*
 * megamol::core::CoreInstance::ViewJobHandleDalloc
 */
void megamol::core::CoreInstance::ViewJobHandleDalloc(void *data,
        megamol::core::ApiHandle *obj) {
    CoreInstance *core = reinterpret_cast<CoreInstance *>(data);
    if (core != NULL) {
        ModuleNamespace *vj = dynamic_cast<ModuleNamespace*>(obj);
        if (vj != NULL) {
            core->closeViewJob(vj);
        }
    }
}


/*
 * megamol::core::CoreInstance::CoreInstance
 */
megamol::core::CoreInstance::CoreInstance(void) : ApiHandle(),
        preInit(new PreInit), config(),
        shaderSourceFactory(config), log(),
        builtinViewDescs(), projViewDescs(), builtinJobDescs(), projJobDescs(),
        pendingViewInstRequests(), pendingJobInstRequests(), namespaceRoot(),
        timeOffset(0.0), plugins(), paramUpdateListeners() {
    //printf("######### PerformanceCounter Frequency %I64u\n", vislib::sys::PerformanceCounter::QueryFrequency());
#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Startup();
#endif /* ULTRA_SOCKET_STARTUP */

    profiler::Manager::Instance().SetCoreInstance(this);

    this->config.instanceLog = &this->log;

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

    // redirect default log to instance log of last instance
    //  not perfect, but better than nothing.
    vislib::sys::Log::DefaultLog.SetLogFileName(
        static_cast<const char*>(NULL), false);
    vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
    vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
    vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::RedirectTarget(&this->log));

    this->log.SetLogFileName(static_cast<const char*>(NULL), false);
    this->log.SetLevel(vislib::sys::Log::LEVEL_ALL);
    this->log.SetEchoLevel(vislib::sys::Log::LEVEL_NONE);
    this->log.SetOfflineMessageBufferSize(25);

    //////////////////////////////////////////////////////////////////////
    // register builtin descriptions
    //////////////////////////////////////////////////////////////////////
    // view descriptions
    //////////////////////////////////////////////////////////////////////
    ViewDescription *vd;

    // empty view; name for compatibility reasons
    vd = new ViewDescription("emptyview");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty View3D
    vd = new ViewDescription("emptyview3d");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty View2D
    vd = new ViewDescription("emptyview2d");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View2D"), "view");
    // 'View2D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // empty view (show the title); name for compatibility reasons
    vd = new ViewDescription("titleview");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // view for powerwall
    vd = new ViewDescription("powerwallview");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("PowerwallView"), "pwview");
    //vd->AddModule(ModuleDescriptionManager::Instance()->Find("ClusterController"), "::cctrl"); // TODO: Dependant instance!
    vd->AddCall(CallDescriptionManager::Instance()->Find("CallRegisterAtController"), "pwview::register", "::cctrl::register");
    vd->SetViewModuleID("pwview");
    this->builtinViewDescs.Register(vd);

    // view for fusionex-hack (client side)
    vd = new ViewDescription("simpleclusterview");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterClient"), "::scc");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterView"), "scview");
    vd->AddCall(CallDescriptionManager::Instance()->Find("SimpleClusterClientViewRegistration"), "scview::register", "::scc::registerView");
    vd->SetViewModuleID("scview");

    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "::logo");
    vd->AddCall(CallDescriptionManager::Instance()->Find("CallRenderView"), "scview::renderView", "::logo::render");

    this->builtinViewDescs.Register(vd);

    vd = new ViewDescription("mpiclusterview");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterClient"), "::mcc");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("MpiClusterView"), "mcview");
    vd->AddCall(CallDescriptionManager::Instance()->Find("SimpleClusterClientViewRegistration"), "mcview::register", "::mcc::registerView");
    vd->SetViewModuleID("mcview");

    this->builtinViewDescs.Register(vd);

    // test view for sphere rendering
    vd = new ViewDescription("testspheres");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleSphereRenderer"), "rnd");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("TestSpheresDataSource"), "dat");
    vd->AddCall(CallDescriptionManager::Instance()->Find("CallRender3D"), "view::rendering", "rnd::rendering");
    vd->AddCall(CallDescriptionManager::Instance()->Find("MultiParticleDataCall"), "rnd::getData", "dat::getData");
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    // test view for sphere rendering
    vd = new ViewDescription("testgeospheres");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleGeoSphereRenderer"), "rnd");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("TestSpheresDataSource"), "dat");
    vd->AddCall(CallDescriptionManager::Instance()->Find("CallRender3D"), "view::rendering", "rnd::rendering");
    vd->AddCall(CallDescriptionManager::Instance()->Find("MultiParticleDataCall"), "rnd::getData", "dat::getData");
    vd->SetViewModuleID("view");
    this->builtinViewDescs.Register(vd);

    //////////////////////////////////////////////////////////////////////
    // job descriptions
    //////////////////////////////////////////////////////////////////////
    JobDescription *jd;

    // job for the cluster controller modules
    jd = new JobDescription("clustercontroller");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("ClusterController"), "::cctrl");
    jd->SetJobModuleID("::cctrl");
    this->builtinJobDescs.Register(jd);

    // job for the cluster controller head-node modules
    jd = new JobDescription("clusterheadcontroller");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("ClusterController"), "::cctrl");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("ClusterViewMaster"), "::cmaster");
    jd->AddCall(CallDescriptionManager::Instance()->Find("CallRegisterAtController"), "::cmaster::register", "::cctrl::register");
    jd->SetJobModuleID("::cctrl");
    this->builtinJobDescs.Register(jd);

    // job for the cluster display client heartbeat server
    jd = new JobDescription("heartbeat");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterClient"), "::scc");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterHeartbeat"), "scheartbeat");
    jd->AddCall(CallDescriptionManager::Instance()->Find("SimpleClusterClientViewRegistration"), "scheartbeat::register", "::scc::registerView");
    jd->SetJobModuleID("scheartbeatthread");
    this->builtinJobDescs.Register(jd);

    // view for fusionex-hack (server side)
    jd = new JobDescription("simpleclusterserver");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("SimpleClusterServer"), "::scs");
    jd->SetJobModuleID("::scs");
    this->builtinJobDescs.Register(jd);

    // // TODO: Replace (is deprecated)
    // job to produce images
    jd = new JobDescription("imagemaker");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("ScreenShooter"), "imgmaker");
    jd->SetJobModuleID("imgmaker");
    this->builtinJobDescs.Register(jd);

    // TODO: Debug
    jd = new JobDescription("DEBUGjob");
    jd->AddModule(ModuleDescriptionManager::Instance()->Find("JobThread"), "ctrl");
    jd->SetJobModuleID("ctrl");
    this->builtinJobDescs.Register(jd);

    //////////////////////////////////////////////////////////////////////

    this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Core Instance created");
}


/*
 * megamol::core::CoreInstance::~CoreInstance
 */
megamol::core::CoreInstance::~CoreInstance(void) {
    this->config.instanceLog = NULL;
    SAFE_DELETE(this->preInit);
    this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Core Instance destroyed");

    // Shutdown all views and jobs, which might still run
    this->namespaceRoot.ModuleGraphLock().LockExclusive();
    AbstractNamedObjectContainer::ChildList remoov;
    AbstractNamedObjectContainer::ChildList::Iterator iter
        = this->namespaceRoot.GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *child = iter.Next();
        if ((dynamic_cast<ViewInstance*>(child) != NULL)
                || (dynamic_cast<JobInstance*>(child) != NULL)) {
            remoov.Add(child);
        }
    }

    iter = remoov.GetIterator();
    while (iter.HasNext()) {
        ModuleNamespace *child = dynamic_cast<ModuleNamespace*>(iter.Next());
        this->closeViewJob(child);
    }
    this->namespaceRoot.ModuleGraphLock().UnlockExclusive();

    ModuleDescriptionManager::ShutdownInstance();
    CallDescriptionManager::ShutdownInstance();

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Cleanup();
#endif /* ULTRA_SOCKET_STARTUP */
}


/*
 * megamol::core::CoreInstance::Initialise
 */
void megamol::core::CoreInstance::Initialise(void) {
    if (this->preInit == NULL) {
        throw vislib::IllegalStateException(
            "Cannot initialise a core instance twice.", __FILE__, __LINE__);
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

    // configuration file
    if (this->preInit->IsConfigFileSet()) {
        this->config.LoadConfig(this->preInit->GetConfigFile());
    } else {
        this->config.LoadConfig();
    }

    // loading plugins
    // printf("Log: %d:\n", (long)(&vislib::sys::Log::DefaultLog));
    // printf("\tAutoflush: %s\n", vislib::sys::Log::DefaultLog.IsAutoFlushEnabled() ? "enabled" : "disabled");
    // printf("\tLevel: %u\n", vislib::sys::Log::DefaultLog.GetLevel());
    // printf("\tEcho-Level: %u\n", vislib::sys::Log::DefaultLog.GetEchoLevel());
    // printf("\tEcho-Target: %d\n", (long)(vislib::sys::Log::DefaultLog.GetEchoOutTarget()));
    vislib::SingleLinkedList<vislib::TString> plugins;
    this->config.ListPluginsToLoad(plugins);
    vislib::SingleLinkedList<vislib::TString>::Iterator iter = plugins.GetIterator();
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
            } catch(...) {
                profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
            }
        }
    } else {
        // Do not profile on default
        profiler::Manager::Instance().SetMode(profiler::Manager::PROFILE_NONE);
    }

    while (this->config.HasInstantiationRequests()) {
        utility::Configuration::InstanceRequest r
            = this->config.GetNextInstantiationRequest();

        megamol::core::ViewDescription *vd = this->FindViewDescription(
            vislib::StringA(r.Description()));
        if (vd != NULL) {
            this->RequestViewInstantiation(vd, r.Identifier(), &r);
            continue;
        }
        megamol::core::JobDescription *jd = this->FindJobDescription(
            vislib::StringA(r.Description()));
        if (jd != NULL) {
            this->RequestJobInstantiation(jd, r.Identifier(), &r);
            continue;
        }

        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Unable to instance \"%s\" as \"%s\": Description not found.\n",
            vislib::StringA(r.Description()).PeekBuffer(),
            vislib::StringA(r.Identifier()).PeekBuffer());
    }

    SAFE_DELETE(this->preInit);

}


/*
 * megamol::core::CoreInstance::SetInitValue
 */
mmcErrorCode megamol::core::CoreInstance::SetInitValue(mmcInitValue key, 
        mmcValueType type, const void* value) {
    if (this->preInit == NULL) {
        throw vislib::IllegalStateException(
            "Core instance already initialised.", __FILE__, __LINE__);
    }

    try {
        switch (key) {
            case MMC_INITVAL_CFGFILE:
                if (!utility::APIValueUtil::IsStringType(type)) {
                    return MMC_ERR_TYPE;
                }
                this->preInit->SetConfigFile(
                    utility::APIValueUtil::AsStringW(type, value));
                break;
            case MMC_INITVAL_CFGSET:
                if (!utility::APIValueUtil::IsStringType(type)) {
                    return MMC_ERR_TYPE;
                }
                this->config.ActivateConfigSet(
                    utility::APIValueUtil::AsStringW(type, value));
                this->log.WriteMsg(250,
                    "Configuration Set \"%s\" added to be activated\n",
                    utility::APIValueUtil::AsStringA(type, value)
                    .PeekBuffer());
                break;
            case MMC_INITVAL_LOGFILE:
                if (!utility::APIValueUtil::IsStringType(type)) {
                    return MMC_ERR_TYPE;
                }
                this->preInit->SetLogFile(
                    utility::APIValueUtil::AsStringW(type, value));
                break;
            case MMC_INITVAL_LOGLEVEL:
                if (!utility::APIValueUtil::IsIntType(type)) {
                    return MMC_ERR_TYPE;
                }
                this->preInit->SetLogLevel(
                    utility::APIValueUtil::AsUint32(type, value));
                break;
            case MMC_INITVAL_LOGECHOLEVEL:
                if (!utility::APIValueUtil::IsIntType(type)) {
                    return MMC_ERR_TYPE;
                }
                this->preInit->SetLogEchoLevel(
                    utility::APIValueUtil::AsUint32(type, value));
                break;
            case MMC_INITVAL_INCOMINGLOG: {
                //if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
                //vislib::sys::Log *log = static_cast<vislib::sys::Log*>(
                //    const_cast<void*>(value));
                //log->SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
                //log->SetEchoTarget(&this->logRedirection);
                //log->EchoOfflineMessages(true);
                //log->SetLogFileName(static_cast<const char*>(NULL), false);
                //log->SetLevel(vislib::sys::Log::LEVEL_NONE);
                return MMC_ERR_NOT_IMPLEMENTED; // use MMC_INITVAL_CORELOG instead
            } break;
            case MMC_INITVAL_CORELOG: {
                if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
                vislib::sys::Log **log = static_cast<vislib::sys::Log**>(
                    const_cast<void*>(value));
                *log = &this->log;
            } break;
            case MMC_INITVAL_LOGECHOFUNC:
                if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
                this->log.SetEchoTarget(new utility::LogEchoTarget(function_cast<mmcLogEchoFunction>(const_cast<void*>(value))));
                break;
            default:
                return MMC_ERR_UNKNOWN;
        }
    } catch(...) {
        return MMC_ERR_UNKNOWN;
    }
    return MMC_ERR_NO_ERROR;
}


/*
 * megamol::core::CoreInstance::FindViewDescription
 */
megamol::core::ViewDescription*
megamol::core::CoreInstance::FindViewDescription(const char *name) {
    ViewDescription *d = NULL;
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
void megamol::core::CoreInstance::EnumViewDescriptions(mmcEnumStringAFunction func, void *data, bool getBuiltinToo)
{
	assert(func);

	auto it = this->projViewDescs.GetIterator();
	while( it.HasNext() ){
		auto vd = it.Next();
		func(vd->ClassName(), data);
	}

	if( getBuiltinToo ) {
		it = this->builtinViewDescs.GetIterator();
		while( it.HasNext() ){
			auto vd = it.Next();
			func(vd->ClassName(), data);
		}
	}

}

/*
 * megamol::core::CoreInstance::FindJobDescription
 */
megamol::core::JobDescription*
megamol::core::CoreInstance::FindJobDescription(const char *name) {
    JobDescription *d = NULL;
    if (d == NULL) {
        d = this->projJobDescs.Find(name);
    }
    if (d == NULL) {
        d = this->builtinJobDescs.Find(name);
    }
    return d;
}


/*
 * megamol::core::CoreInstance::RequestAllInstantiations
 */
void megamol::core::CoreInstance::RequestAllInstantiations() {
    auto itervd = this->projViewDescs.GetIterator();
    while (itervd.HasNext()) {
        int cnt = this->pendingViewInstRequests.Count();
        std::string s = std::to_string(cnt);
        vislib::StringA name = "v";
        name.Append(s.c_str());
        ViewDescription *vd = itervd.Next();
        ViewInstanceRequest req;
        req.SetName(name);
        req.SetDescription(vd);
        this->pendingViewInstRequests.Add(req);
    }
    auto iterjd = this->projJobDescs.GetIterator();
    while (iterjd.HasNext()) {
        int cnt = this->pendingJobInstRequests.Count();
        std::string s = std::to_string(cnt);
        vislib::StringA name = "j";
        name.Append(s.c_str());
        JobDescription *jd = iterjd.Next();
        JobInstanceRequest req;
        req.SetName(name);
        req.SetDescription(jd);
        this->pendingJobInstRequests.Add(req);
    }
}


/*
 * megamol::core::CoreInstance::RequestViewInstantiation
 */
void megamol::core::CoreInstance::RequestViewInstantiation(
        megamol::core::ViewDescription *desc, const vislib::StringA& id,
        const ParamValueSetRequest *param) {
    if (id.Find(':') != vislib::StringA::INVALID_POS) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "View instantiation request aborted: name contains invalid character \":\"");
        return;
    }
    // could check here if the description is instantiable, but I do not want
    // to.
    ASSERT(desc != NULL);
    ViewInstanceRequest req;
    req.SetName(id);
    req.SetDescription(desc);
    if (param != NULL) {
        static_cast<ParamValueSetRequest&>(req) = *param;
    }
    this->pendingViewInstRequests.Add(req);
}


/*
 * megamol::core::CoreInstance::RequestJobInstantiation
 */
void megamol::core::CoreInstance::RequestJobInstantiation(
        megamol::core::JobDescription *desc, const vislib::StringA& id,
        const ParamValueSetRequest *param) {
    if (id.Find(':') != vislib::StringA::INVALID_POS) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Job instantiation request aborted: name contains invalid character \":\"");
        return;
    }
    // could check here if the description is instantiable, but I do not want
    // to.
    ASSERT(desc != NULL);
    JobInstanceRequest req;
    req.SetName(id);
    req.SetDescription(desc);
    if (param != NULL) {
        static_cast<ParamValueSetRequest&>(req) = *param;
    }
    this->pendingJobInstRequests.Add(req);
}


/*
 * megamol::core::CoreInstance::InstantiatePendingView
 */
megamol::core::ViewInstance *
megamol::core::CoreInstance::InstantiatePendingView(void) {
    using vislib::sys::Log;
    VLSTACKTRACE("InstantiatePendingView", __FILE__, __LINE__);

    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, true);
    vislib::sys::AutoLock lock(locker);

    if (this->pendingViewInstRequests.IsEmpty()) return NULL;

    ViewInstanceRequest request
        = this->pendingViewInstRequests.First();
    this->pendingViewInstRequests.RemoveFirst();

    ModuleNamespace *preViewInst = NULL;
    bool hasErrors = false;
    view::AbstractView *view = NULL, *fallbackView = NULL;
    vislib::StringA viewFullPath
        = this->namespaceRoot.FullNamespace(request.Name(), request.Description()->ViewModuleID());

    AbstractNamedObject *ano = this->namespaceRoot.FindChild(request.Name());
    if (ano != NULL) {
        preViewInst = dynamic_cast<ModuleNamespace*>(ano);
        if (preViewInst == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate view %s: non-namespace object blocking instance name\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    } else {
        // thomasbm: das hier erzeugt memory leaks! Wem soll das neue 'ModuleNamespace' gehoeren bzw. wer ist fuers deleten zustaendig?
        preViewInst = new ModuleNamespace(request.Name());
        this->namespaceRoot.AddChild(preViewInst);
    }

    if (preViewInst == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate view %s: Internal Error %d\n",
            request.Name().PeekBuffer(), __LINE__);
        return NULL;
    }

    { // glh "fix"
        vislib::Array<vislib::StringA> exts(vislib::StringTokeniserA::Split(
            vislib::StringA(reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS))), ' ', true));
        for (SIZE_T i = 0; i < exts.Count(); i++) {
            glh_init_extension(exts[i]);
        }
    }

    // instantiate modules
    for (unsigned int idx = 0; idx < request.Description()->ModuleCount(); idx++) {
        const ViewDescription::ModuleInstanceRequest &mir = request.Description()->Module(idx);
        ModuleDescription *desc = mir.Second();

        vislib::StringA fullName = this->namespaceRoot.FullNamespace(request.Name(), mir.First());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n", fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module *mod = this->instantiateModule(fullName, desc);
        if (mod == NULL) {
            hasErrors = true;
            continue;

        } else {
            view::AbstractView* av = dynamic_cast<view::AbstractView*>(mod);
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
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate view %s: No view module found\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < request.Description()->CallCount(); idx++) {
        const ViewDescription::CallInstanceRequest &cir = request.Description()->Call(idx);
        CallDescription *desc = cir.Description();

        vislib::StringA fromFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.From());
        vislib::StringA toFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.To());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL)) profiler::Manager::Instance().Select(fromFullName);
        }
    }

    if (hasErrors) {
        this->CleanupModuleGraph();

    } else {
        // Create Instance object replacing the temporary namespace
        ViewInstance *inst = new ViewInstance();
        if (inst == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to construct instance %s\n",
                request.Name().PeekBuffer());
            return NULL;
        }

        if (!inst->Initialize(preViewInst, view)) {
            SAFE_DELETE(inst);
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to initialize instance %s\n",
                request.Name().PeekBuffer());
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
megamol::core::view::AbstractView *
megamol::core::CoreInstance::instantiateSubView(megamol::core::ViewDescription *vd) {
    using vislib::sys::Log;
    VLSTACKTRACE("instantiateSubView", __FILE__, __LINE__);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, true);
    vislib::sys::AutoLock lock(locker);

    bool hasErrors = false;
    view::AbstractView *view = NULL, *fallbackView = NULL;

    // instantiate modules
    for (unsigned int idx = 0; idx < vd->ModuleCount(); idx++) {
        const ViewDescription::ModuleInstanceRequest &mir = vd->Module(idx);
        ModuleDescription *desc = mir.Second();
        const vislib::StringA& fullName = mir.First();

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n", fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module *mod = this->instantiateModule(fullName, desc);
        if (mod == NULL) {
            hasErrors = true;
            continue;

        } else {
            view::AbstractView* av = dynamic_cast<view::AbstractView*>(mod);
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
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate view %s: No view module found\n",
                vd->ClassName());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < vd->CallCount(); idx++) {
        const ViewDescription::CallInstanceRequest &cir = vd->Call(idx);
        CallDescription *desc = cir.Description();
        const vislib::StringA& fromFullName = cir.From();
        const vislib::StringA& toFullName = cir.To();

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL)) profiler::Manager::Instance().Select(fromFullName);
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
megamol::core::JobInstance *
megamol::core::CoreInstance::InstantiatePendingJob(void) {
    using vislib::sys::Log;
    VLSTACKTRACE("InstantiatePendingJob", __FILE__, __LINE__);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, true);
    vislib::sys::AutoLock lock(locker);

    if (this->pendingJobInstRequests.IsEmpty()) return NULL;

    JobInstanceRequest request = this->pendingJobInstRequests.First();
    this->pendingJobInstRequests.RemoveFirst();

    ModuleNamespace *preJobInst = NULL;
    bool hasErrors = false;
    job::AbstractJob *job = NULL, *fallbackJob = NULL;
    vislib::StringA jobFullPath
        = this->namespaceRoot.FullNamespace(request.Name(), request.Description()->JobModuleID());

    AbstractNamedObject *ano = this->namespaceRoot.FindChild(request.Name());
    if (ano != NULL) {
        preJobInst = dynamic_cast<ModuleNamespace*>(ano);
        if (preJobInst == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate job %s: non-namespace object blocking instance name\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    } else {
        preJobInst = new ModuleNamespace(request.Name());
        this->namespaceRoot.AddChild(preJobInst);
    }

    if (preJobInst == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate job %s: Internal Error %d\n",
            request.Name().PeekBuffer(), __LINE__);
        return NULL;
    }

    // instantiate modules
    for (unsigned int idx = 0; idx < request.Description()->ModuleCount(); idx++) {
        const JobDescription::ModuleInstanceRequest &mir = request.Description()->Module(idx);
        ModuleDescription *desc = mir.Second();

        vislib::StringA fullName = this->namespaceRoot.FullNamespace(request.Name(), mir.First());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate module \"%s\": request data corrupted "
                "due invalid module class name.\n", fullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Module *mod = this->instantiateModule(fullName, desc);
        if (mod == NULL) {
            hasErrors = true;
            continue;

        } else {
            job::AbstractJob* aj = dynamic_cast<job::AbstractJob*>(mod);
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
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate job %s: No job controller module found\n",
                request.Name().PeekBuffer());
            return NULL;
        }
    }

    // instantiate calls
    for (unsigned int idx = 0; idx < request.Description()->CallCount(); idx++) {
        const JobDescription::CallInstanceRequest &cir = request.Description()->Call(idx);
        CallDescription *desc = cir.Description();

        vislib::StringA fromFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.From());
        vislib::StringA toFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.To());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->InstantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
        } else if (profiler::Manager::Instance().GetMode() != profiler::Manager::PROFILE_NONE) {
            if (cir.DoProfiling() || (profiler::Manager::Instance().GetMode() == profiler::Manager::PROFILE_ALL)) profiler::Manager::Instance().Select(fromFullName);
        }
    }

    if (hasErrors) {
        this->CleanupModuleGraph();

    } else {
        // Create Instance object replacing the temporary namespace
        JobInstance *inst = new JobInstance();
        if (inst == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to construct instance %s\n",
                request.Name().PeekBuffer());
            return NULL;
        }

        if (!inst->Initialize(preJobInst, job)) {
            SAFE_DELETE(inst);
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to initialize instance %s\n",
                request.Name().PeekBuffer());
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
vislib::SmartPtr<megamol::core::param::AbstractParam>
megamol::core::CoreInstance::FindParameterIndirect(const vislib::StringA& name, bool quiet)
{
	vislib::StringA paramName(name);
	vislib::SmartPtr<core::param::AbstractParam> param;
	vislib::SmartPtr<core::param::AbstractParam> lastParam;
	while ((param = this->FindParameter(paramName, quiet)) != nullptr)
	{
		lastParam = param;
		paramName = param->ValueString();
	}
	return lastParam;
}

/*
 * megamol::core::CoreInstance::FindParameter
 */
vislib::SmartPtr<megamol::core::param::AbstractParam>
megamol::core::CoreInstance::FindParameter(const vislib::StringA& name, bool quiet, bool create) {
    using vislib::sys::Log;
    VLSTACKTRACE("FindParameter", __FILE__, __LINE__);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, false);
    vislib::sys::AutoLock lock(locker);

    vislib::Array<vislib::StringA> path = vislib::StringTokeniserA::Split(name, "::", true);
	vislib::StringA slotName("");
	if (path.Count() > 0)
	{
		slotName = path.Last();
		path.RemoveLast();
	}
	vislib::StringA modName("");
	if (path.Count() > 0)
	{
		modName = path.Last();
		path.RemoveLast();
	}

    ModuleNamespace *mn = NULL;
    // parameter slots may have namespace operators in their names!
    while (mn == NULL) {
        mn = this->namespaceRoot.FindNamespace(path, false, true);
        if (mn == NULL) {
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
					if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
						"Cannot find parameter \"%s\": namespace not found",
						name.PeekBuffer());
					return NULL;
				}
            }
        }
    }

    Module *mod = dynamic_cast<Module *>(mn->FindChild(modName));
    if (mod == NULL) {
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
			if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
				"Cannot find parameter \"%s\": module not found",
				name.PeekBuffer());
	        return NULL;
		}
    }

    param::ParamSlot *slot = dynamic_cast<param::ParamSlot*>(mod->FindChild(slotName));
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
			if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
				"Cannot find parameter \"%s\": slot not found",
				name.PeekBuffer());
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
			if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
				"Cannot find parameter \"%s\": slot is not available",
				name.PeekBuffer());
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
			if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
				"Cannot find parameter \"%s\": slot has no parameter",
				name.PeekBuffer());
			return NULL;
		}
    }



    return slot->Parameter();
}


/*
 * megamol::core::CoreInstance::LoadProject
 */
void megamol::core::CoreInstance::LoadProject(const vislib::StringA& filename) {
    megamol::core::utility::xml::XmlReader reader;
    if (!reader.OpenFile(filename)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to open project file \"%s\"",
            filename.PeekBuffer());
        return;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Loading project file \"%s\"", filename.PeekBuffer());
    this->addProject(reader);
}


/*
 * megamol::core::CoreInstance::LoadProject
 */
void megamol::core::CoreInstance::LoadProject(const vislib::StringW& filename) {
    megamol::core::utility::xml::XmlReader reader;
    if (!reader.OpenFile(filename)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to open project file \"%s\"",
            vislib::StringA(filename).PeekBuffer());
        return;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Loading project file \"%s\"",
        vislib::StringA(filename).PeekBuffer());
    this->addProject(reader);
}


/*
 * megamol::core::CoreInstance::GetInstanceTime
 */
double megamol::core::CoreInstance::GetCoreInstanceTime(void) const {
//#ifdef _WIN32 // not a good idea when running more than one megamol on one machine
//    ::SetThreadAffinityMask(::GetCurrentThread(), 0x00000001);
//#endif /* _WIN32 */
    return vislib::sys::PerformanceCounter::QueryMillis()
        * 0.001 + this->timeOffset;
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
    VLSTACKTRACE("CleanupModuleGraph", __FILE__, __LINE__);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, true);
    vislib::sys::AutoLock lock(locker);

    this->namespaceRoot.SetAllCleanupMarks();

    AbstractNamedObjectContainer::ChildList::Iterator iter
        = this->namespaceRoot.GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *child = iter.Next();
        ViewInstance *vi = dynamic_cast<ViewInstance*>(child);
        JobInstance *ji = dynamic_cast<JobInstance*>(child);

        if (vi != NULL) {
            vi->ClearCleanupMark();
        }
        if (ji != NULL) {
            ji->ClearCleanupMark();
        }
    }

    this->namespaceRoot.DisconnectCalls();
    this->namespaceRoot.PerformCleanup();

}


/*
 * megamol::core::CoreInstance::Shutdown
 */
void megamol::core::CoreInstance::Shutdown(void) {
    AbstractNamedObjectContainer::ChildList::Iterator iter
        = this->namespaceRoot.GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *child = iter.Next();
        ViewInstance *vi = dynamic_cast<ViewInstance*>(child);
        JobInstance *ji = dynamic_cast<JobInstance*>(child);

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
void megamol::core::CoreInstance::SetupGraphFromNetwork(const void * data) {
    using vislib::sys::Log;
    using vislib::net::AbstractSimpleMessage;

    const AbstractSimpleMessage *dataPtr
        = static_cast<const AbstractSimpleMessage *>(data);
    const AbstractSimpleMessage &dat = *dataPtr;

    this->namespaceRoot.ModuleGraphLock().LockExclusive();
    try {

        UINT64 cntMods = *dat.GetBodyAs<UINT64>();
        UINT64 cntCalls = *dat.GetBodyAsAt<UINT64>(sizeof(UINT64));
        UINT64 cntParams = *dat.GetBodyAsAt<UINT64>(sizeof(UINT64) * 2);
        SIZE_T pos = sizeof(UINT64) * 3;

        //printf("\n\nGraph Setup:\n");
        for (UINT64 i = 0; i < cntMods; i++) {
            vislib::StringA modClass(dat.GetBodyAsAt<char>(pos));
            pos += modClass.Length() + 1;
            vislib::StringA modName(dat.GetBodyAsAt<char>(pos));
            pos += modName.Length() + 1;

            if (modClass.Equals(cluster::ClusterViewMaster::ClassName())
                    || modClass.Equals(cluster::ClusterController::ClassName())
                    || modClass.Equals(cluster::simple::Server::ClassName())) {
                // these are infra structure modules and not to be synced
                continue;
            }

            ModuleDescription *d = ModuleDescriptionManager::Instance()->Find(modClass);
            if (d == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s(%s): class description not found\n",
                    modName.PeekBuffer(), modClass.PeekBuffer());
                continue;
            }
            Module *m = this->instantiateModule(modName, d);
            if (m == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s(%s): instantiation failed\n",
                    modName.PeekBuffer(), modClass.PeekBuffer());
                continue;
            }
            //printf("    Modul: %s as %s\n", modClass.PeekBuffer(), modName.PeekBuffer());

        }

        for (UINT64 i = 0; i < cntCalls; i++) {
            vislib::StringA callClass(dat.GetBodyAsAt<char>(pos));
            pos += callClass.Length() + 1;
            vislib::StringA callFrom(dat.GetBodyAsAt<char>(pos));
            pos += callFrom.Length() + 1;
            vislib::StringA callTo(dat.GetBodyAsAt<char>(pos));
            pos += callTo.Length() + 1;

            AbstractNamedObject *ano = this->namespaceRoot.FindNamedObject(callFrom);
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Not connecting call %s from missing module %s\n",
                    callClass.PeekBuffer(), callFrom.PeekBuffer());
                continue;
            }
            ano = this->namespaceRoot.FindNamedObject(callTo);
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Not connecting call %s to missing module %s\n",
                    callClass.PeekBuffer(), callTo.PeekBuffer());
                continue;
            }

            CallDescription *d = CallDescriptionManager::Instance()->Find(callClass);
            if (d == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate call %s=>%s(%s): class description not found\n",
                    callFrom.PeekBuffer(), callTo.PeekBuffer(), callClass.PeekBuffer());
                continue;
            }
            Call *c = this->InstantiateCall(callFrom, callTo, d);
            if (c == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to instantiate module %s=>%s(%s): instantiation failed\n",
                    callFrom.PeekBuffer(), callTo.PeekBuffer(), callClass.PeekBuffer());
                continue;
            }
            //printf("    Call: %s from %s to %s\n", callClass.PeekBuffer(), callFrom.PeekBuffer(), callTo.PeekBuffer());
        }

        for (UINT64 i = 0; i < cntParams; i++) {
            vislib::StringA paramName(dat.GetBodyAsAt<char>(pos));
            pos += paramName.Length() + 1;
            vislib::StringA paramValue(dat.GetBodyAsAt<char>(pos));
            pos += paramValue.Length() + 1;

            AbstractNamedObject *ano = this->namespaceRoot.FindNamedObject(paramName);
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 10, // could be a warning, but we intentionally omitted some modules
                    "Parameter %s not found\n", paramName.PeekBuffer());
                continue;
            }
            param::ParamSlot *ps = dynamic_cast<param::ParamSlot*>(ano);
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
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to decode parameter value for %s\n", paramName.PeekBuffer());
                continue;
            }
            ps->Parameter()->ParseValue(value);
            //printf("    Param: %s to %s\n", paramName.PeekBuffer(), paramValue.PeekBuffer());
        }
        //printf("\n");

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to setup module graph from network message: %s\n",
            ex.GetMsgA());
    } catch(...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to setup module graph from network message: unexpected exception\n");
    }
    this->namespaceRoot.ModuleGraphLock().UnlockExclusive();
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
    unsigned char *buffer = new unsigned char[maxBufferSize];

    vislib::sys::File file;
    if (!file.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
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
    ModuleDescription *dataSrcClass = NULL;
    ModuleDescriptionManager::DescriptionIterator di = ModuleDescriptionManager::Instance()->GetIterator();

    // first try auto-detect with loaders with matching file name extension
    while (di.HasNext()) {
        ModuleDescription *md = di.Next();
        if (!md->IsLoaderWithAutoDetection()) continue;
        const char *extsStr = md->LoaderAutoDetectionFilenameExtensions();
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
        di = ModuleDescriptionManager::Instance()->GetIterator();
        while (di.HasNext()) {
            ModuleDescription *md = di.Next();
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
        Log::DefaultLog.WriteError(_T("Failed to Quickstart \"%s\": No suitable data source class found"), filename.PeekBuffer());
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
        di = ModuleDescriptionManager::Instance()->GetIterator();
        while (di.HasNext()) {
            ModuleDescription *md = di.Next();
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
                Log::DefaultLog.WriteWarn("Unable to connect data source with configured renderer for quickstart. Trying auto-search");
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
            Log::DefaultLog.WriteError(_T("Failed to Quickstart \"%s\": Cannot auto-connect data source to view"), filename.PeekBuffer());
            return;
        }
        ASSERT(!view.ViewModuleID().IsEmpty());
    }

    ViewDescription *newview = new ViewDescription(view.ClassName());
    Log::DefaultLog.WriteInfo(10, "Quickstart module graph with %u modules and %u calls defined:",
            view.ModuleCount(), view.CallCount());
    for (unsigned int i = 0; i < view.ModuleCount(); i++) {
        Log::DefaultLog.WriteInfo(25, "Module \"%s\" of class \"%s\"\n",
            view.Module(i).First().PeekBuffer(), view.Module(i).Second()->ClassName());
        newview->AddModule(view.Module(i).Second(), view.Module(i).First());
    }
    for (unsigned int i = 0; i < view.CallCount(); i++) {
        Log::DefaultLog.WriteInfo(25, "Call from \"%s\" to \"%s\" of class \"%s\"\n",
            view.Call(i).From().PeekBuffer(),
            view.Call(i).To().PeekBuffer(),
            view.Call(i).Description()->ClassName());
        newview->AddCall(view.Call(i).Description(), view.Call(i).From(), view.Call(i).To());
    }
    for (unsigned int i = 0; i < view.ParamValueCount(); i++) {
        newview->AddParamValue(view.ParamValue(i).First(), view.ParamValue(i).Second());
    }
    newview->SetViewModuleID(view.ViewModuleID());
    this->builtinViewDescs.Register(newview);

    viewName.Format("q%d", quickstartCounter);
    this->RequestViewInstantiation(newview, viewName);
    quickstartCounter++;
    Log::DefaultLog.WriteInfo("Quickstart view instantiation request posted");

}


/*
 * megamol::core::CoreInstance::QuickstartRegistry
 */
void megamol::core::CoreInstance::QuickstartRegistry(const vislib::TString& frontend,
        const vislib::TString& feparams, const vislib::TString& filetype, bool unreg, bool overwrite) {
    using vislib::sys::Log;
    vislib::TString filetypename;

    if (filetype.IsEmpty()) {
        Log::DefaultLog.WriteError("Empty file type is illegal\n");
        return;
    }
    if (filetype.Equals(_T("*"))) {
        // all file types
        ModuleDescriptionManager::DescriptionIterator di
            = ModuleDescriptionManager::Instance()->GetIterator();
        while (di.HasNext()) {
            const ModuleDescription *md = di.Next();
            if (!md->IsVisibleForQuickstart()) continue;
            const char *fnextsstr = md->LoaderAutoDetectionFilenameExtensions();
            const char *fnnamestr = md->LoaderFileTypeName();
            if (fnextsstr == NULL) continue;
            vislib::Array<vislib::TString> fnexts = vislib::TStringTokeniser::Split(A2T(fnextsstr), _T(";"), true);
            for (SIZE_T i = 0; i < fnexts.Count(); i++) {
                filetypename.Format(_T("%s File"), 
                    ((fnnamestr == NULL)
                        ? fnexts[i].Substring(1)
                        : vislib::TString(fnnamestr)).PeekBuffer());
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
    ModuleDescriptionManager::DescriptionIterator di
        = ModuleDescriptionManager::Instance()->GetIterator();
    while (di.HasNext()) {
        const ModuleDescription *md = di.Next();
        if (!md->IsVisibleForQuickstart()) continue;
        const char *fnextsstr = md->LoaderAutoDetectionFilenameExtensions();
        if (fnextsstr == NULL) continue;
        vislib::Array<vislib::TString> fnexts = vislib::TStringTokeniser::Split(A2T(fnextsstr), _T(";"), true);
        for (SIZE_T i = 0; i < fnexts.Count(); i++) {
            if (fnexts[i].Equals(fnext, false)) {
                const char *fnnamestr = md->LoaderFileTypeName();
                filetypename.Format(_T("%s File"), 
                    ((fnnamestr == NULL)
                        ? fnext.Substring(1)
                        : vislib::TString(fnnamestr)).PeekBuffer());
                if (unreg) {
                    this->unregisterQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
                } else {
                    this->registerQuickstart(frontend, feparams, fnext, filetypename, !overwrite);
                }
                return;
            }
        }
    }

    Log::DefaultLog.WriteWarn(_T("Quickstart %sregistration for unknown type %s"),
        ((unreg) ? _T("un") : _T("")), filetype.PeekBuffer());
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
bool megamol::core::CoreInstance::WriteStateToXML(const char *outFilename) {
    using namespace vislib::sys;

    BufferedFile outfile;
    if (!outfile.Open(outFilename, File::WRITE_ONLY, File::SHARE_READ,
            File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to create state file file.");
        return false;
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "State has been written to '%s'",
                outFilename);
    }

    // Write root tag and 'header'
    WriteLineToFile(outfile, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    WriteLineToFile(outfile, "<!--\n");
    WriteLineToFile(outfile, "This file has been auto-generated by MegaMol.\n");
    WriteLineToFile(outfile, "-->\n");
    WriteLineToFile(outfile, "<MegaMol type=\"project\" version=\"1.0\">\n");

    // Loop through active view instances
    // collect data
    vislib::Stack<AbstractNamedObject *> stack;
    stack.Push(&this->namespaceRoot);
    auto itervd = this->projViewDescs.GetIterator();
    auto iterjd = this->projJobDescs.GetIterator();

    int nViews = 1;
    int nJobs = 1;
    vislib::StringA closeModuleTags = "";
    vislib::StringA closeViewJobTags = "";
    vislib::StringA callDesc = "";
    while (!stack.IsEmpty()) {
        AbstractNamedObject *ano = stack.Pop();
        ASSERT(ano != NULL);
        AbstractNamedObjectContainer *anoc =
                dynamic_cast<AbstractNamedObjectContainer *>(ano);
        Module *mod = dynamic_cast<Module *>(ano);
        //CalleeSlot *callee = dynamic_cast<CalleeSlot *>(ano);
        CallerSlot *caller = dynamic_cast<CallerSlot *>(ano);
        param::ParamSlot *param = dynamic_cast<param::ParamSlot *>(ano);

        megamol::core::ViewInstance *vi = dynamic_cast<megamol::core::ViewInstance *>(ano);
        megamol::core::JobInstance *ji = dynamic_cast<megamol::core::JobInstance *>(ano);

        if (anoc != NULL) {
//            printf("NAMED OBJECT: %s\n", anoc->FullName().PeekBuffer());
            AbstractNamedObjectContainer::ChildList::Iterator anoccli =
                    anoc->GetChildIterator();
            while (anoccli.HasNext()) {
                stack.Push(anoccli.Next());
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
            Module *mod = dynamic_cast<Module*>(ji->Job());
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

            ModuleDescription *d = NULL;
            ModuleDescriptionManager::DescriptionIterator i =
                    ModuleDescriptionManager::Instance()->GetIterator();
            while (i.HasNext()) {
                ModuleDescription *id = i.Next();
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


            Call *c = caller->CallAs<Call>();
            if (c == NULL)
                continue;
            CallDescription *d = NULL;
            CallDescriptionManager::DescriptionIterator i =
                    CallDescriptionManager::Instance()->GetIterator();
            while (i.HasNext()) {
                CallDescription *id = i.Next();
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
            if (param->Parameter().IsNull())
                continue;
//            printf("        PARAM %s = %s\n", param->FullName().PeekBuffer(),
//                    param->Parameter()->ValueString().PeekBuffer());

            // Write button parameters commented out, since we do not want
            // to trigger them by default
            if (param->Param<param::ButtonParam>() != NULL) {
                WriteLineToFile(outfile, "      <!--<param name=\"");
                WriteLineToFile(outfile, param->FullName().PeekBuffer());
                WriteLineToFile(outfile, "\" value=\"");
                WriteLineToFile(outfile,
                        param->Parameter()->ValueString().PeekBuffer());
                WriteLineToFile(outfile, "\" />-->\n");
            } else {
                WriteLineToFile(outfile, "      <param name=\"");
                WriteLineToFile(outfile, param->FullName().PeekBuffer());
                WriteLineToFile(outfile, "\" value=\"");
                WriteLineToFile(outfile,
                        param->Parameter()->ValueString().PeekBuffer());
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
 * megamol::core::CoreInstance::addProject
 */
void megamol::core::CoreInstance::addProject(
        megamol::core::utility::xml::XmlReader& reader) {
    using vislib::sys::Log;
    utility::ProjectParser parser;
    if (parser.Parse(reader)) {
        // success, add project elements
        ViewDescription *vd = NULL;
        while (true) {
            vd = parser.PopViewDescription();
            if (vd != NULL) {
                this->projViewDescs.Register(vd);
            } else break;
        }
        JobDescription *jd = NULL;
        while (true) {
            jd = parser.PopJobDescription();
            if (jd != NULL) {
                this->projJobDescs.Register(jd);
            } else break;
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
void debugDumpSlots(megamol::core::AbstractNamedObjectContainer *c) {
    using vislib::sys::Log;
    using megamol::core::AbstractNamedObject;
    using megamol::core::AbstractNamedObjectContainer;
    using megamol::core::CalleeSlot;
    using megamol::core::CallerSlot;
    using megamol::core::param::ParamSlot;
    vislib::SingleLinkedList<megamol::core::AbstractNamedObject*>::Iterator iter = c->GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *ano = iter.Next();
        AbstractNamedObjectContainer *anoc = dynamic_cast<AbstractNamedObjectContainer *>(ano);
        CalleeSlot *callee = dynamic_cast<CalleeSlot *>(ano);
        CallerSlot *caller = dynamic_cast<CallerSlot *>(ano);
        ParamSlot *param = dynamic_cast<ParamSlot *>(ano);

        if (callee != NULL) {
            Log::DefaultLog.WriteInfo(150, "Callee Slot: %s", callee->FullName().PeekBuffer());
        } else if (caller != NULL) {
            Log::DefaultLog.WriteInfo(150, "Caller Slot: %s", caller->FullName().PeekBuffer());
        } else if (param != NULL) {
            Log::DefaultLog.WriteInfo(150, "Param Slot: %s", param->FullName().PeekBuffer());
        } else if (anoc != NULL) {
            debugDumpSlots(anoc);
        }
    }
}

#endif /* DEBUG || _DEBUG */


/*
 * megamol::core::CoreInstance::instantiateModule
 */
megamol::core::Module* megamol::core::CoreInstance::instantiateModule(
        const vislib::StringA path, ModuleDescription* desc) {
    using vislib::sys::Log;
    VLSTACKTRACE("instantiateModule", __FILE__, __LINE__);

#if defined(DEBUG) || defined(_DEBUG)

    unsigned int plc = 0;
    mmcParamSlotDescription *pl = NULL;
    unsigned int celc = 0;
    mmcCalleeSlotDescription *cel = NULL;
    unsigned int crlc = 0;
    mmcCallerSlotDescription *crl = NULL;

    ::mmcGetModuleSlotDescriptions(static_cast<void*>(desc),
        &plc, &pl, &celc, &cel, &crlc, &crl);

    ::mmcReleaseModuleSlotDescriptions(
        plc, &pl, celc, &cel, crlc, &crl);

#endif /* DEBUG */

    ASSERT(path.StartsWith("::"));
    ASSERT(desc != NULL);

    if (!desc->IsAvailable()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to make module \"%s\" (%s): Module type is installed by not available.",
            desc->ClassName(), path.PeekBuffer());
        return NULL;
    }

    vislib::Array<vislib::StringA> dirs = vislib::StringTokeniserA::Split(path, "::", true);
    vislib::StringA modName = dirs.Last();
    dirs.RemoveLast();

    ModuleNamespace *cns = this->namespaceRoot.FindNamespace(dirs, true);
    if (cns == NULL) {
        return NULL;
    }

    AbstractNamedObject *ano = cns->FindChild(modName);
    if (ano != NULL) {
        Module *tstMod = dynamic_cast<Module*>(ano);
        if ((tstMod != NULL) && (desc->IsDescribing(tstMod))) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Unable to make module \"%s\": module already present",
                path.PeekBuffer());
            return tstMod;
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to make module \"%s\" (%s): name conflict with other namespace object.",
            desc->ClassName(), path.PeekBuffer());
        return NULL;
    }

    Module *mod = desc->CreateModule(modName, this);
    if (mod == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to construct module \"%s\" (%s)",
            desc->ClassName(), path.PeekBuffer());

    } else if (!mod->Create()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to create module \"%s\" (%s)",
            desc->ClassName(), path.PeekBuffer());
        SAFE_DELETE(mod);

    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350,
            "Created module \"%s\" (%s)",
            desc->ClassName(), path.PeekBuffer());
        cns->AddChild(mod);
#if defined(DEBUG) || defined(_DEBUG)
        debugDumpSlots(mod);
#endif /* DEBUG || _DEBUG */

    }

    return mod;
}


/*
 * megamol::core::CoreInstance::InstantiateCall
 */
megamol::core::Call* megamol::core::CoreInstance::InstantiateCall(
        const vislib::StringA fromPath, const vislib::StringA toPath,
        megamol::core::CallDescription* desc) {
    using vislib::sys::Log;
    VLSTACKTRACE("InstantiateCall", __FILE__, __LINE__);

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

    ModuleNamespace *fromNS = this->namespaceRoot.FindNamespace(fromDirs, false);
    if (fromNS == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find source namespace \"%s\"",
            fromPath.PeekBuffer());
        return NULL;
    }

    ModuleNamespace *toNS = this->namespaceRoot.FindNamespace(toDirs, false);

    if (toNS == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find target namespace \"%s\"",
            toPath.PeekBuffer());
        return NULL;
    }

    Module *fromMod = dynamic_cast<Module*>(fromNS->FindChild(fromModName));
    if (fromMod == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find source module \"%s\"",
            fromPath.PeekBuffer());
        return NULL;
    }

    Module *toMod = dynamic_cast<Module*>(toNS->FindChild(toModName));
    if (toMod == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find target module \"%s\"",
            toPath.PeekBuffer());
        return NULL;
    }

    CallerSlot *fromSlot = dynamic_cast<CallerSlot*>(fromMod->FindSlot(fromSlotName));
    if (fromSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find source slot \"%s\"",
            fromPath.PeekBuffer());
        return NULL;
    }

    CalleeSlot *toSlot = dynamic_cast<CalleeSlot*>(toMod->FindSlot(toSlotName));
    if (toSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: can not find target slot \"%s\"",
            toPath.PeekBuffer());
        return NULL;
    }

    if (!fromSlot->IsCallCompatible(desc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: source slot \"%s\" not compatible with call \"%s\"",
            fromPath.PeekBuffer(), desc->ClassName());
        return NULL;
    }

    if (!toSlot->IsCallCompatible(desc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: target slot \"%s\" not compatible with call \"%s\"",
            toPath.PeekBuffer(), desc->ClassName());
        return NULL;
    }

    if ((fromSlot->GetStatus() == AbstractSlot::STATUS_CONNECTED)
            && (toSlot->GetStatus() == AbstractSlot::STATUS_CONNECTED)) {
        Call *tstCall = fromSlot->IsConnectedTo(toSlot);
        if (tstCall != NULL) {
            if (desc->IsDescribing(tstCall)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Unable to instantiate call \"%s\"->\"%s\": call already exists",
                    fromPath.PeekBuffer(), toPath.PeekBuffer());
                return tstCall;
            }
        }
    }

    if (fromSlot->GetStatus() != AbstractSlot::STATUS_ENABLED) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: status of source slot \"%s\" is invalid.",
            fromPath.PeekBuffer());
        return NULL;
    }

    Call *call = desc->CreateCall();
    if (call == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: failed to create call \"%s\"",
            desc->ClassName());
        return NULL;
    }

    if (!toSlot->ConnectCall(call)) {
        delete call;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: failed to connect call \"%s\" to target slot \"%s\"",
            desc->ClassName(), toPath.PeekBuffer());
        return NULL;
    }

    if (!fromSlot->ConnectCall(call)) {
        delete call; // Disconnects call as sfx
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate call: failed to connect call \"%s\" to source slot \"%s\"",
            desc->ClassName(), fromPath.PeekBuffer());
        return NULL;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350,
        "Call \"%s\" instantiated from \"%s\" to \"%s\"",
        desc->ClassName(), fromPath.PeekBuffer(), toPath.PeekBuffer());

    return call;
}


/*
 * megamol::core::CoreInstance::enumParameters
 */
void megamol::core::CoreInstance::enumParameters(
        const megamol::core::ModuleNamespace* path,
        mmcEnumStringAFunction func, void *data) const {
    VLSTACKTRACE("enumParameters", __FILE__, __LINE__);

    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, false);
    vislib::sys::AutoLock lock(locker);

    vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> i
        = path->GetConstChildIterator();
    while (i.HasNext()) {
        const AbstractNamedObject *child = i.Next();
        const Module *mod = dynamic_cast<const Module*>(child);
        const ModuleNamespace *ns = dynamic_cast<const ModuleNamespace*>(child);

        if (mod != NULL) {
            vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> si
                = mod->GetConstChildIterator();
            while (si.HasNext()) {
                const param::ParamSlot *slot = dynamic_cast<const param::ParamSlot*>(si.Next());
                if (slot != NULL) {
                    vislib::StringA name(mod->FullName());
                    name.Append("::");
                    name.Append(slot->Name());
                    func(name.PeekBuffer(), data);
                }
            }

        } else if (ns != NULL) {
            this->enumParameters(ns, func, data);
        }
    }
}


/*
 * megamol::core::CoreInstance::findParameterName
 */
vislib::StringA megamol::core::CoreInstance::findParameterName(
        const megamol::core::ModuleNamespace* path,
        const vislib::SmartPtr<megamol::core::param::AbstractParam>& param)
        const {
    VLSTACKTRACE("findParameterName", __FILE__, __LINE__);

    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, false);
    vislib::sys::AutoLock lock(locker);

    vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> i
        = path->GetConstChildIterator();
    while (i.HasNext()) {
        const AbstractNamedObject *child = i.Next();
        const Module *mod = dynamic_cast<const Module*>(child);
        const ModuleNamespace *ns = dynamic_cast<const ModuleNamespace*>(child);

        if (mod != NULL) {
            vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> si
                = mod->GetConstChildIterator();
            while (si.HasNext()) {
                const param::ParamSlot *slot = dynamic_cast<const param::ParamSlot*>(si.Next());
                if ((slot != NULL) && (slot->Parameter() == param)) {
                    vislib::StringA name(mod->FullName());
                    name.Append("::");
                    name.Append(slot->Name());
                    return name;
                }
            }

        } else if (ns != NULL) {
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
void megamol::core::CoreInstance::closeViewJob(megamol::core::ModuleNamespace *obj) {
    VLSTACKTRACE("closeViewJob", __FILE__, __LINE__);

    ASSERT(obj != NULL);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, true);
    vislib::sys::AutoLock lock(locker);

    if (obj->Parent() != &this->namespaceRoot) {
        // this happens when a job/view is removed from the graph before it's
        // handle is deletes (core instance destructor)
        return;
    }

    ViewInstance *vi = dynamic_cast<ViewInstance*>(obj);
    JobInstance *ji = dynamic_cast<JobInstance*>(obj);
    if (vi != NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 50,
            "View instance %s terminating ...", vi->Name().PeekBuffer());
        vi->Terminate();
    }
    if (ji != NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 50,
            "Job instance %s terminating ...", ji->Name().PeekBuffer());
        ji->Terminate();
    }

    ModuleNamespace *mn = new ModuleNamespace(obj->Name());

    AbstractNamedObjectContainer::ChildList::Iterator iter = obj->GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *ano = iter.Next();
        obj->RemoveChild(ano);
        mn->AddChild(ano);
    }

    AbstractNamedObjectContainer *p = dynamic_cast<AbstractNamedObjectContainer *>(obj->Parent());
    p->RemoveChild(obj);
    p->AddChild(mn);

    ASSERT(obj->Parent() == NULL);
    ASSERT(!obj->GetChildIterator().HasNext());
    // obj will be deleted outside this method

    this->CleanupModuleGraph();
}


/*
 * megamol::core::CoreInstance::applyConfigParams
 */
void megamol::core::CoreInstance::applyConfigParams(
        const vislib::StringA& root, const InstanceDescription *id,
        const ParamValueSetRequest *params) {
    VLSTACKTRACE("applyConfigParams", __FILE__, __LINE__);

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
                "Initializing parameter \"%s\" to \"%s\"",
                nameA.PeekBuffer(), vislib::StringA(pvr.Second()).PeekBuffer());
            p->ParseValue(pvr.Second());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "Unable to set parameter \"%s\" to \"%s\": parameter not found",
                nameA.PeekBuffer(), vislib::StringA(pvr.Second()).PeekBuffer());
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
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Setting parameter \"%s\" to \"%s\"",
                nameA.PeekBuffer(), vislib::StringA(pvr.Second()).PeekBuffer());
            p->ParseValue(pvr.Second());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "Unable to set parameter \"%s\" to \"%s\": parameter not found",
                nameA.PeekBuffer(), vislib::StringA(pvr.Second()).PeekBuffer());
        }
    }

}


/*
 * megamol::core::CoreInstance::loadPlugin
 */
void megamol::core::CoreInstance::loadPlugin(const vislib::TString &filename) {
    vislib::sys::DynamicLinkLibrary *plugin = new vislib::sys::DynamicLinkLibrary();
    unsigned int loadFailedLevel = vislib::sys::Log::LEVEL_ERROR;
    if (this->config.IsConfigValueSet("PluginLoadFailMsg")) {
        try {
            const vislib::StringW &v = this->config.ConfigValue("PluginLoadFailMsg");
            if (v.Equals(L"error", false) || v.Equals(L"err", false) || v.Equals(L"e", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_ERROR;
            } else 
            if (v.Equals(L"warning", false) || v.Equals(L"warn", false) || v.Equals(L"w", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_WARN;
            } else 
            if (v.Equals(L"information", false) || v.Equals(L"info", false) || v.Equals(L"i", false)
                    || v.Equals(L"message", false) || v.Equals(L"msg", false) || v.Equals(L"m", false)) {
                loadFailedLevel = vislib::sys::Log::LEVEL_INFO;
            } else {
                loadFailedLevel = vislib::CharTraitsW::ParseInt(v.PeekBuffer());
            }
        } catch(...) {
        }
    }

    // load module
    if (!plugin->Load(filename)) {
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": Cannot load plugin \"%s\"\n",
            vislib::StringA(filename).PeekBuffer(),
            plugin->LastLoadErrorMessage().PeekBuffer());
        delete plugin;
        return;
    }

    // test api version
    int (*mmplgPluginAPIVersion)(void) = function_cast<int (*)()>(plugin->GetProcAddress("mmplgPluginAPIVersion"));
    if ((mmplgPluginAPIVersion == NULL)) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": API entry not found\n",
            vislib::StringA(filename).PeekBuffer());
        return;
    }
    int plgApiVer = mmplgPluginAPIVersion();
    if (plgApiVer != 100) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": incompatible API version\n",
            vislib::StringA(filename).PeekBuffer());
        return;
    }

    // test core compatibility
    const void* (*mmplgCoreCompatibilityValue)(void) = function_cast<const void* (*)()>(plugin->GetProcAddress("mmplgCoreCompatibilityValue"));
    if ((mmplgCoreCompatibilityValue == NULL)) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": API function \"mmplgCoreCompatibilityValue\" not found\n",
            vislib::StringA(filename).PeekBuffer());
        return;
    }
    const mmplgCompatibilityValues *compVal = static_cast<const mmplgCompatibilityValues*>(mmplgCoreCompatibilityValue());
    if ((compVal->size != sizeof(mmplgCompatibilityValues)) 
            || (compVal->mmcoreRev != MEGAMOL_CORE_COMP_REV)) {
        SIZE_T rev = compVal->mmcoreRev;
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": core version mismatch (%d from Core; %d from Plugin)\n",
            vislib::StringA(filename).PeekBuffer(),
            static_cast<int>(MEGAMOL_CORE_COMP_REV),
            static_cast<int>(rev));
        return;
    }
    if ((compVal->size != sizeof(mmplgCompatibilityValues)) 
            || ((compVal->vislibRev != 0) && (compVal->vislibRev != VISLIB_VERSION_REVISION))) {
        SIZE_T rev = compVal->vislibRev;
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": vislib version mismatch (%d from Core; %d from Plugin)\n",
            vislib::StringA(filename).PeekBuffer(),
            static_cast<int>(VISLIB_VERSION_REVISION),
            static_cast<int>(rev));
        return;
    }

    // connect static objects
    bool (*mmplgConnectStatics)(int, void*) = function_cast<bool (*)(int, void*)>(plugin->GetProcAddress("mmplgConnectStatics"));
    if (mmplgConnectStatics != NULL) {
        bool rv;
        rv = mmplgConnectStatics(1, static_cast<void*>(&vislib::sys::Log::DefaultLog));
        VLTRACE(VISLIB_TRCELVL_INFO, "Plug-in connect log: %s\n", rv ? "true" : "false");
        vislib::SmartPtr<vislib::StackTrace> stackManager(vislib::StackTrace::Manager());
        rv = mmplgConnectStatics(2, static_cast<void*>(&stackManager));
        VLTRACE(VISLIB_TRCELVL_INFO, "Plug-in connect stacktrace: %s\n", rv ? "true" : "false");
    }

    // load description
    const char * (*mmplgPluginName)(void) = function_cast<const char * (*)()>(plugin->GetProcAddress("mmplgPluginName"));
//    const char * (*mmplgPluginDescription)(void) = function_cast<const char * (*)()>(plugin->GetProcAddress("mmplgPluginDescription"));
    if ((mmplgPluginName == NULL)/* || (mmplgPluginDescription == NULL)*/) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": API name/description functions not found\n",
            vislib::StringA(filename).PeekBuffer());
        return;
    }
    vislib::StringA plgName(mmplgPluginName());
    if (plgName.IsEmpty()) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": Plugin does not export a name\n",
            vislib::StringA(filename).PeekBuffer());
        return;
    }

    // object export information
    int (*mmplgModuleCount)(void) = function_cast<int (*)()>(plugin->GetProcAddress("mmplgModuleCount"));
    void* (*mmplgModuleDescription)(int) = function_cast<void* (*)(int)>(plugin->GetProcAddress("mmplgModuleDescription"));

    int (*mmplgCallCount)(void) = function_cast<int (*)()>(plugin->GetProcAddress("mmplgCallCount"));
    void* (*mmplgCallDescription)(int) = function_cast<void* (*)(int)>(plugin->GetProcAddress("mmplgCallDescription"));

    int modCnt = ((mmplgModuleCount == NULL) || (mmplgModuleDescription == NULL)) ? 0 : mmplgModuleCount();
    int modCntVal = 0;
    for (int i = 0; i < modCnt; i++) {
        void * modPtr = mmplgModuleDescription(i);
        if (modPtr == NULL) continue;
        ModuleDescriptionManager::Instance()->Register(
            reinterpret_cast<ModuleDescription *>(modPtr));
        modCntVal++;
    }

    int callCnt = ((mmplgCallCount == NULL) || (mmplgCallDescription == NULL)) ? 0 : mmplgCallCount();
    int callCntVal = 0;
    for (int i = 0; i < callCnt; i++) {
        void * callPtr = mmplgCallDescription(i);
        if (callPtr == NULL) continue;
        CallDescriptionManager::Instance()->Register(
            reinterpret_cast<CallDescription *>(callPtr));
        callCntVal++;
    }

    this->log.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Plugin \"%s\" loaded: %d Modules, %d Calls registered\n",
        vislib::StringA(filename).PeekBuffer(), modCntVal, callCntVal);

    this->plugins.Add(plugin);
}


/*
 * megamol::core::CoreInstance::quickConnectUp
 */
bool megamol::core::CoreInstance::quickConnectUp(megamol::core::ViewDescription& view, const char *from, const char *to) {
    using vislib::sys::Log;

    vislib::SingleLinkedList<vislib::Array<quickStepInfo> > fifo;
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

    while(!fifo.IsEmpty()) {
        vislib::Array<quickStepInfo> list(fifo.First());
        fifo.RemoveFirst();
        const quickStepInfo& lqsi = list.Last();
        ASSERT(lqsi.nextMod != NULL);

        //printf("Test connection from %s\n", lqsi.nextMod->ClassName());
        this->quickConnectUpStepInfo(lqsi.nextMod, connInfo);

        // test for end condition
        if (to == NULL) {
            for (SIZE_T i = 0; i < connInfo.Count(); i++) {
                if (vislib::StringA("View2D").Equals(connInfo[i].nextMod->ClassName(), false)
                        || vislib::StringA("View3D").Equals(connInfo[i].nextMod->ClassName(), false)) {

                    vislib::StringA prevModName(from);
                    for (SIZE_T j = 1; j < list.Count(); j++) {
                        vislib::StringA modName;
                        modName.Format("mod%.3d", view.ModuleCount());
                        view.AddModule(list[j].nextMod, modName);
                        view.AddCall(list[j].call,
                            modName + "::" + list[j].nextSlot,
                            prevModName + "::" + list[j].prevSlot);
                        prevModName = modName;
                    }

                    view.AddModule(connInfo[i].nextMod, "view");
                    view.AddCall(connInfo[i].call,
                        vislib::StringA("view::") + connInfo[i].nextSlot,
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
                        view.AddCall(list[j].call,
                            modName + "::" + list[j].nextSlot,
                            prevModName + "::" + list[j].prevSlot);
                        prevModName = modName;
                    }

                    view.AddCall(connInfo[i].call,
                        vislib::StringA(to) + "::" + connInfo[i].nextSlot,
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
void megamol::core::CoreInstance::quickConnectUpStepInfo(megamol::core::ModuleDescription *from,
        vislib::Array<megamol::core::CoreInstance::quickStepInfo>& step) {
    using vislib::sys::Log;
    ASSERT(from != NULL);
    step.Clear();

    Module *m = from->CreateModule("quickstarttest", this);
    if (m == NULL) {
        Log::DefaultLog.WriteError("Unable to test-instantiate module %s", from->ClassName());
        return;
    }
    vislib::SingleLinkedList<vislib::Pair<CallDescription*, vislib::StringA> > inCalls;
    vislib::Stack<AbstractNamedObjectContainer *> stack(m);
    while (!stack.IsEmpty()) {
        AbstractNamedObjectContainer *anoc = stack.Pop();
        AbstractNamedObjectContainer::ChildList::Iterator ci = anoc->GetChildIterator();
        while (ci.HasNext()) {
            AbstractNamedObject *ano = ci.Next();
            if (dynamic_cast<AbstractNamedObjectContainer*>(ano) != NULL) {
                stack.Push(dynamic_cast<AbstractNamedObjectContainer*>(ano));
            }
            if (dynamic_cast<CalleeSlot*>(ano) != NULL) {
                CalleeSlot *callee = dynamic_cast<CalleeSlot*>(ano);

                CallDescriptionManager::DescriptionIterator cdi = CallDescriptionManager::Instance()->GetIterator();
                while (cdi.HasNext()) {
                    CallDescription *cd = cdi.Next();
                    vislib::Pair<CallDescription*, vislib::StringA> cse(cd, callee->Name());
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
    delete m;
    // now 'inCalls' holds all calls which can connect to 'from' (some slot)

    ModuleDescriptionManager::DescriptionIterator mdi = ModuleDescriptionManager::Instance()->GetIterator();
    while (mdi.HasNext()) {
        ModuleDescription *md = mdi.Next();
        if (md == from) continue;
        if (!md->IsVisibleForQuickstart()) continue;
        m = md->CreateModule("quickstarttest", this);
        if (m == NULL) continue;

        bool connectable = false;
        stack.Push(m);
        while (!stack.IsEmpty() && !connectable) {
            AbstractNamedObjectContainer *anoc = stack.Pop();
            ASSERT(anoc != NULL);
            AbstractNamedObjectContainer::ChildList::Iterator ci = anoc->GetChildIterator();
            while (ci.HasNext() && !connectable) {
                AbstractNamedObject *ano = ci.Next();
                ASSERT(ano != NULL);
                if (dynamic_cast<AbstractNamedObjectContainer*>(ano) != NULL) {
                    stack.Push(dynamic_cast<AbstractNamedObjectContainer*>(ano));
                }
                if (dynamic_cast<CallerSlot*>(ano) != NULL) {
                    CallerSlot *caller = dynamic_cast<CallerSlot*>(ano);
                    ASSERT(caller != NULL);

                    vislib::SingleLinkedList<vislib::Pair<CallDescription*, vislib::StringA> >::Iterator cdi = inCalls.GetIterator();
                    while (cdi.HasNext() && !connectable) {
                        const vislib::Pair<CallDescription*, vislib::StringA> &cde = cdi.Next();
                        CallDescription *cd = cde.First();
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
        delete m;
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
        const vislib::TString& fnext, const vislib::TString& fnname,
        bool keepothers) {
    using vislib::sys::Log;
#ifdef _WIN32
    using vislib::sys::RegistryKey;
#endif /* _WIN32 */
    ASSERT(!fnext.IsEmpty());
    ASSERT(fnext[0] == _T('.'));
#ifdef _WIN32
    Log::DefaultLog.WriteInfo(_T("Registering \"%s\" type (*%s) for quickstart"), fnname.PeekBuffer(), fnext.PeekBuffer());
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
            typeName.Format(_T("MegaMol.%d.%d%s"), MEGAMOL_PRODUCT_MAJOR_VER, MEGAMOL_PRODUCT_MINOR_VER, fnext.PeekBuffer());
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
#if defined(DEBUG) || defined(_DEBUG)
        str.Append(_T(" [Debug]"));
#endif /* DEBUG || _DEBUG */
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

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError(_T("Cannot register quickstart for %s: %s (%s, %d)"), fnext.PeekBuffer(),
            ex.GetMsg(), vislib::TString(ex.GetFile()).PeekBuffer(), ex.GetLine());
    } catch(...) {
        Log::DefaultLog.WriteError(_T("Cannot register quickstart for %s: Unexpected Exception"), fnext.PeekBuffer());
    }
#else /* _WIN32 */
    Log::DefaultLog.WriteError(_T("Quickstart registration is not supported on this operating system"), fnext.PeekBuffer());
#endif /* _WIN32 */
}


/*
 * megamol::core::CoreInstance::unregisterQuickstart
 */
void megamol::core::CoreInstance::unregisterQuickstart(const vislib::TString& frontend, const vislib::TString& feparams,
        const vislib::TString& fnext, const vislib::TString& fnname,
        bool keepothers) {
    using vislib::sys::Log;
#ifdef _WIN32
    using vislib::sys::RegistryKey;
#endif /* _WIN32 */
    ASSERT(!fnext.IsEmpty());
    ASSERT(fnext[0] == _T('.'));
#ifdef _WIN32
    Log::DefaultLog.WriteInfo(_T("Un-Registering \"%s\" type (*%s) for quickstart"), fnname.PeekBuffer(), fnext.PeekBuffer());
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
            Log::DefaultLog.WriteWarn(_T("File type %s does not seem to be correctly registered (#0x1)"), fnext.PeekBuffer());
            if (extKey.GetSubKeysA().Count() == 0) {
                extKey.Close();
                crw.DeleteSubKey(fnext); // just delete it already, since it's broken anyway
            }
            return;
        }

        RegistryKey typeKey;
        errcode = crw.OpenSubKey(typeKey, typeName);
        if (errcode != ERROR_SUCCESS) {
            Log::DefaultLog.WriteWarn(_T("File type %s does not seem to be correctly registered (#0x2)"), fnext.PeekBuffer());
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
#if defined(UNICODE) || defined(_UNICODE)
                    shell.GetSubKeysW();
#else /* UNICODE || _UNICODE */
                    shell.GetSubKeysA();
#endif /* UNICODE || _UNICODE */
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
                Log::DefaultLog.WriteWarn(_T("File type %s does not seem to be correctly registered (#0x3)"), fnext.PeekBuffer());
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

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError(_T("Cannot unregister quickstart for %s: %s (%s, %d)"), fnext.PeekBuffer(),
            ex.GetMsg(), vislib::TString(ex.GetFile()).PeekBuffer(), ex.GetLine());
    } catch(...) {
        Log::DefaultLog.WriteError(_T("Cannot unregister quickstart for %s: Unexpected Exception"), fnext.PeekBuffer());
    }
#else /* _WIN32 */
    Log::DefaultLog.WriteWarn(_T("Quickstart registration is not supported on this operating system"), fnext.PeekBuffer());
#endif /* _WIN32 */
}
