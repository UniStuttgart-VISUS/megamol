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
#include "Module.h"
#include "ModuleDescription.h"
#include "ModuleDescriptionManager.h"
#include "param/ParamSlot.h"
#include "utility/APIValueUtil.h"
#include "utility/ProjectParser.h"
#include "utility/xml/XmlReader.h"
#include "job/JobThread.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include "vislib/functioncast.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/GUID.h"
#include "vislib/Socket.h"
#include "vislib/StackTrace.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/NetworkInformation.h"
#include "vislib/vislibversion.h"


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
        shaderSourceFactory(config), log(), logRedirection(), logEchoTarget(),
        builtinViewDescs(), projViewDescs(), builtinJobDescs(),
        pendingViewInstRequests(), pendingJobInstRequests(), namespaceRoot(),
        timeOffset(0.0), plugins() {

#ifdef ULTRA_SOCKET_STARTUP
    vislib::net::Socket::Startup();
#endif /* ULTRA_SOCKET_STARTUP */

    this->config.instanceLog = &this->log;
    this->logRedirection.SetTarget(&this->log);

    // Normalize timer with time offset to something less crappy shitty hateworthy
    this->timeOffset = -this->GetInstanceTime();

#ifdef _DEBUG
    // Use a randomized time offset to debug the offset synchronization
    { // use a guid to initialize the pseudo-random generator
        vislib::GUID guid;
        guid.Create();
        ::srand(guid.HashCode());
    }
    this->timeOffset += 100.0 * static_cast<double>(::rand()) / static_cast<double>(RAND_MAX);
#endif

    // redirect default log to instance log of last instance
    //  not perfect, but better than nothing.
    vislib::sys::Log::DefaultLog.SetLogFileName(
        static_cast<const char*>(NULL), false);
    vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
    vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
    vislib::sys::Log::DefaultLog.SetEchoOutTarget(&this->logRedirection);

    this->log.SetLogFileName(static_cast<const char*>(NULL), false);
    this->log.SetLevel(vislib::sys::Log::LEVEL_ALL);
    this->log.SetEchoLevel(vislib::sys::Log::LEVEL_NONE);
    this->log.SetOfflineMessageBufferSize(25);
    this->log.SetEchoOutTarget(&this->logEchoTarget);

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

    // empty View3D
    vd = new ViewDescription("emptyview3d");
    vd->AddModule(ModuleDescriptionManager::Instance()->Find("View3D"), "view");
    // 'View3D' will show the title logo as long as no renderer is connected
    vd->SetViewModuleID("view");

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

    //// DEBUG! TODO: Remove
    //vd->AddModule(ModuleDescriptionManager::Instance()->Find("View2D"), "view");
    //vd->AddModule(ModuleDescriptionManager::Instance()->Find("ChronoGraph"), "watch");
    //vd->AddCall(CallDescriptionManager::Instance()->Find("CallRender2D"), "view::rendering", "watch::rendering");
    //vd->AddCall(CallDescriptionManager::Instance()->Find("CallRenderView"), "pwview::renderView", "view::render");

    vd->SetViewModuleID("pwview");
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

    // // TODO: Replace (is deprecated)
    //jd = new JobDescription("imagemaker");
    //jd->AddModule(ModuleDescriptionManager::Instance()->Find("ScreenShooter"), "imgmaker");
    //jd->SetJobModuleID("imgmaker");
    //this->builtinJobDescs.Register(jd);

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
    this->namespaceRoot.LockModuleGraph(true);
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
    this->namespaceRoot.UnlockModuleGraph();

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
            this->log.EchoOfflineMessages(false);
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
                if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
                vislib::sys::Log *log = static_cast<vislib::sys::Log*>(
                    const_cast<void*>(value));
                log->SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
                log->SetEchoOutTarget(&this->logRedirection);
                log->EchoOfflineMessages(true);
                log->SetLogFileName(static_cast<const char*>(NULL), false);
                log->SetLevel(vislib::sys::Log::LEVEL_NONE);
            } break;
            case MMC_INITVAL_LOGECHOFUNC:
                if (type != MMC_TYPE_VOIDP) { return MMC_ERR_TYPE; }
                this->logEchoTarget.SetTarget(
                    function_cast<mmcLogEchoFunction>(
                        const_cast<void*>(value)));
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
 * megamol::core::CoreInstance::FindJobDescription
 */
megamol::core::JobDescription*
megamol::core::CoreInstance::FindJobDescription(const char *name) {
    JobDescription *d = NULL;
    // TODO: Search in project jobs
    if (d == NULL) {
        d = this->builtinJobDescs.Find(name);
    }
    return d;
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
        preViewInst = new ModuleNamespace(request.Name());
        this->namespaceRoot.AddChild(preViewInst);
    }

    if (preViewInst == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to instantiate view %s: Internal Error %d\n",
            request.Name().PeekBuffer(), __LINE__);
        return NULL;
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
        CallDescription *desc = cir.Second();

        vislib::StringA fromFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.First().First());
        vislib::StringA toFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.First().Second());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->instantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
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
        CallDescription *desc = cir.Second();
        const vislib::StringA& fromFullName = cir.First().First();
        const vislib::StringA& toFullName = cir.First().Second();

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->instantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
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
        CallDescription *desc = cir.Second();

        vislib::StringA fromFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.First().First());
        vislib::StringA toFullName = this->namespaceRoot.FullNamespace(request.Name(), cir.First().Second());

        if (desc == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to instantiate call \"%s\"=>\"%s\": request data corrupted "
                "due invalid call class name.\n",
                fromFullName.PeekBuffer(), toFullName.PeekBuffer());
            hasErrors = true;
            continue;
        }

        Call *call = this->instantiateCall(fromFullName, toFullName, desc);
        if (call == NULL) {
            hasErrors = true;
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
megamol::core::CoreInstance::FindParameter(const vislib::StringA& name, bool quiet) {
    using vislib::sys::Log;
    VLSTACKTRACE("FindParameter", __FILE__, __LINE__);
    AbstractNamedObject::GraphLocker locker(&this->namespaceRoot, false);
    vislib::sys::AutoLock lock(locker);

    vislib::Array<vislib::StringA> path = vislib::StringTokeniserA::Split(name, "::", true);
    vislib::StringA slotName = path.Last();
    path.RemoveLast();
    vislib::StringA modName = path.Last();
    path.RemoveLast();

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
                if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Cannot find parameter \"%s\": namespace not found",
                    name.PeekBuffer());
                return NULL;
            }
        }
    }

    Module *mod = dynamic_cast<Module *>(mn->FindChild(modName));
    if (mod == NULL) {
        if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter \"%s\": module not found",
            name.PeekBuffer());
        return NULL;
    }

    param::ParamSlot *slot = dynamic_cast<param::ParamSlot*>(mod->FindChild(slotName));
    if (slot == NULL) {
        if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter \"%s\": slot not found",
            name.PeekBuffer());
        return NULL;
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) {
        if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter \"%s\": slot is not available",
            name.PeekBuffer());
        return NULL;
    }
    if (slot->Parameter().IsNull()) {
        if (!quiet) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter \"%s\": slot has no parameter",
            name.PeekBuffer());
        return NULL;
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
double megamol::core::CoreInstance::GetInstanceTime(void) const {
    return vislib::sys::PerformanceCounter::QueryMillis()
        * 0.001 + this->timeOffset;
}


/*
 * megamol::core::CoreInstance::OffsetInstanceTime
 */
void megamol::core::CoreInstance::OffsetInstanceTime(double offset) {
    this->timeOffset += offset;
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
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to parse project file:");
        vislib::SingleLinkedList<vislib::StringA>::Iterator msgs = parser.Messages();
        while (msgs.HasNext()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s\n", msgs.Next().PeekBuffer());
        }
    }
}


/*
 * megamol::core::CoreInstance::instantiateModule
 */
megamol::core::Module* megamol::core::CoreInstance::instantiateModule(
        const vislib::StringA path, ModuleDescription* desc) {
    using vislib::sys::Log;
    VLSTACKTRACE("instantiateModule", __FILE__, __LINE__);

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
        cns->AddChild(mod);
    }

    return mod;
}


/*
 * megamol::core::CoreInstance::instantiateCall
 */
megamol::core::Call* megamol::core::CoreInstance::instantiateCall(
        const vislib::StringA fromPath, const vislib::StringA toPath,
        megamol::core::CallDescription* desc) {
    using vislib::sys::Log;
    VLSTACKTRACE("instantiateCall", __FILE__, __LINE__);

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
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": Cannot load plugin\n",
            vislib::StringA(filename).PeekBuffer());
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
            || (compVal->mmcoreRev != MEGAMOL_CORE_COMP_REV)
            || ((compVal->vislibRev != 0) && (compVal->vislibRev != VISLIB_VERSION_REVISION))) {
        plugin->Free();
        delete plugin;
        this->log.WriteMsg(loadFailedLevel,
            "Unable to load Plugin \"%s\": core version mismatch\n",
            vislib::StringA(filename).PeekBuffer());
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

}
