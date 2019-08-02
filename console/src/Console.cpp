/*
 * Console.cpp
 *
 * Copyright (C) 2006 - 2016 by MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */

#include "stdafx.h"
#include <cstdlib>
#include <climits>
#include <ctime>
#include <signal.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <thread>

#include "mmcore/api/MegaMolCore.h"

#include "utility/AboutInfo.h"
#include "utility/CmdLineParser.h"
#include "utility/ConfigHelper.h"
#include "CoreHandle.h"
#ifdef _WIN32
#include "utility/WindowsUtils.h"
#endif /* _WIN32 */
#include "JobManager.h"
#include "WindowManager.h"
#include "utility/ParamFileManager.h"

#include "vislib/sys/BufferedFile.h"
#include "vislib/functioncast.h"
#include "vislib/String.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/Trace.h"

/** The core instance handle. */
static megamol::console::CoreHandle hCore;

/** An application wide termination request */
static bool terminationRequest;

/**
 * Flag if the log echo printed an important message (which will change the
 * application return value to nonzero).
 */
static bool echoedImportant;

/** The path to the front end executable currently executing */
static vislib::TString applicationExecutablePath;

#ifndef _WIN32
static void(*oldSignl)(int);
#endif /* !_WIN32 */

/*
 * forward declarations
 */
namespace {

int runNormal(megamol::console::utility::CmdLineParser *& parser);
void signalCtrlC(int);
void MEGAMOLCORE_CALLBACK writeLogEchoToConsole(unsigned int level, const char* message);

void initTraceAndLog();
void echoCmdLine(vislib::sys::TCmdLineProvider& cmdline);
void printErrorsAndWarnings(megamol::console::utility::CmdLineParser* parser);
void loadCmdFromFile(megamol::console::utility::CmdLineParser* parser, vislib::sys::TCmdLineProvider& cmdline, char** argv, int& retVal);
void countProjects(vislib::SingleLinkedList<vislib::TString>& projects, int& loadedProjects, int& loadedLuaProjects);

}


/**
 * The applications main entry point
 *
 * @param argc The number of arguments.
 * @param argv The arguments.
 *
 * @return The return value of the application.
 */
#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[]) {
#if (defined(DEBUG) || defined(_DEBUG)) && defined(_CRTDBG_MAP_ALLOC)
    _CrtSetDbgFlag(
        _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF// | _CRTDBG_CHECK_ALWAYS_DF
        | _CRTDBG_CHECK_CRT_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_CHECK_EVERY_128_DF);
#if defined(MY_CRTDBG_BREAK_AT_ALLOC)
    _CrtSetBreakAlloc(MY_CRTDBG_BREAK_AT_ALLOC);
#endif /*  */
#endif /* DEBUG || _DEBUG */
#else /* _WIN32 */
int main(int argc, char* argv[]) {
#endif /* _WIN32 */
    int retVal = 0;
    megamol::console::utility::CmdLineParser *parser = NULL;

    // Global Variables
    ::terminationRequest = false;
    ::echoedImportant = false;
    ::applicationExecutablePath = argv[0];
    ::initTraceAndLog();

    try {
        vislib::sys::TCmdLineProvider cmdline(argc, argv);
        parser = new megamol::console::utility::CmdLineParser();

        // Parse the command line
        retVal = parser->Parse(cmdline.ArgC(), cmdline.ArgV());

        if (parser->NoConColour()) {
            vislib::sys::Console::EnableColors(false);
        }

        if (retVal < 0) {
            // errors are present!
            ::printErrorsAndWarnings(parser);

        } else {
            // ok or warnings!

            if (parser->UseCmdLineFile()) {
                ::loadCmdFromFile(parser, cmdline, argv, retVal);
            }

            if (!parser->HideLogo() || parser->ShowVersionInfo()) {
                megamol::console::utility::AboutInfo::PrintGreeting();
            }

            if (parser->EchoCmdLine()) {
                ::echoCmdLine(cmdline);
            }

            if (parser->ShowVersionInfo()) {
                megamol::console::utility::AboutInfo::PrintVersionInfo();
            }

            if (retVal > 0) {
                // warnings are present
                parser->PrintErrorsAndWarnings(stderr);
            }

            // call the required 'run*' method
            if (retVal >= 0) {
                if (parser->WantHelp()) {
                    // online help
                    parser->PrintHelp();

                } else {
                    // normal operations
                    retVal = runNormal(parser);

                }
            }
        }

    } catch(vislib::Exception e) {
        vislib::StringA msg;
        msg.Format("Exception in (%s, %d): %s\n", e.GetFile(), e.GetLine(), e.GetMsgA());
        vislib::sys::Log::DefaultLog.WriteError(msg);
        retVal = -1;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Exception.");
        retVal = -1;
    }

    if (parser != NULL) {
        delete parser;
        parser = NULL;
    }

    ::hCore.DestroyHandle();

    // we're done
    if (::echoedImportant && (retVal == 0)) retVal = 1;
    std::cout << std::endl;
#if defined(_WIN32) && defined(_DEBUG) // VC Debugger Halt on Stop Crowbar
#pragma warning(disable: 4996)
    if (/*(retVal != 0) && */(getenv("STOP") != nullptr)) system("pause");
#pragma warning(default: 4996)
#endif
    return retVal;
}


namespace {

void initTraceAndLog() {
    // VISlib TRACE
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL);

    // VISlib Log
    vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
    vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_ALL);
    vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
    vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::StreamTarget(stdout, vislib::sys::Log::LEVEL_ALL));
    megamol::console::utility::AboutInfo::LogGreeting();
    megamol::console::utility::AboutInfo::LogVersionInfo();
    megamol::console::utility::AboutInfo::LogStartTime();
}

void printErrorsAndWarnings(megamol::console::utility::CmdLineParser* parser) {
    megamol::console::utility::AboutInfo::PrintGreeting();
    parser->PrintErrorsAndWarnings(stderr);
    std::cerr << std::endl << "Use \"--help\" for usage information" << std::endl << std::endl;
}

void loadCmdFromFile(megamol::console::utility::CmdLineParser* parser, vislib::sys::TCmdLineProvider& cmdline, char** argv, int& retVal) {
    const TCHAR* cmdlinefile = parser->CmdLineFile();
    if (cmdlinefile == NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to retreive command line file name.\n");

    } else {
        vislib::TString cmdlinefilename(cmdlinefile);
        vislib::sys::BufferedFile file;
        if (file.Open(cmdlinefilename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
                vislib::sys::File::OPEN_ONLY)) {
            vislib::StringA newCmdLine = vislib::sys::ReadLineFromFileA(file, 100000);
            file.Close();

            cmdline.CreateCmdLine(argv[0], A2T(newCmdLine));
            retVal = parser->Parse(cmdline.ArgC(), cmdline.ArgV());

            if (retVal < 0) {
                ::printErrorsAndWarnings(parser);

            } else {
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Read command line from \"%s\"\n", vislib::StringA(cmdlinefilename).PeekBuffer());
            }

        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to open file \"%s\"\n", vislib::StringA(cmdlinefilename).PeekBuffer());
        }
    }
}

void echoCmdLine(vislib::sys::TCmdLineProvider& cmdline) {
    // echo the command line
    vislib::StringA cmdlineecho("Called:");
    for (int i = 0; i < cmdline.ArgC(); i++) {
        cmdlineecho.Append(" ");
        cmdlineecho.Append(vislib::StringA(cmdline.ArgV()[i]));
    }

    vislib::sys::Log::DefaultLog.WriteInfo("%s\n", cmdlineecho.PeekBuffer());
    if (vislib::sys::Log::DefaultLog.GetEchoLevel() < vislib::sys::Log::LEVEL_INFO) {
        std::cout << cmdlineecho << std::endl;
    }
}

/**
 * Creates the core instance object.
 *
 * @return true on success, false on failure.
 */
bool createCoreInstance() {
    mmcErrorCode ccrv = ::mmcCreateCore(hCore);
    if (ccrv == MMC_ERR_NO_ERROR) return true;

    std::cerr << "Unable to create core instance handle:" << std::endl;
    switch (ccrv) {
    case MMC_ERR_MEMORY:
        std::cerr << "\tFailed to allocate instance memory." << std::endl;
        break;
    case MMC_ERR_HANDLE:
        std::cerr << "\tInvalid handle pointer encountered." << std::endl;
        break;
    case MMC_ERR_LICENSING:
        std::cerr << "\tNo valid license found." << std::endl;
        break;
    case MMC_ERR_UNKNOWN:
        std::cerr << "\tUnknown internal error encountered." << std::endl;
        break;
    default:
        std::cerr << "\tUnexpected error code returned." << std::endl;
        break;
    }
    return false;
}

/**
 * Sets up the core instance settings based on parser input
 *
 * @param parser The command line parser
 *
 * @throws vislib::exception on errors
 */
void setupCore(megamol::console::utility::CmdLineParser *& parser) {
    MMC_USING_VERIFY;

    // log

    //MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, // is now deprecated
    //    MMC_INITVAL_INCOMINGLOG, MMC_TYPE_VOIDP, 
    //    static_cast<void*>(&Log::DefaultLog)));
    // HAZARD!!! Cross-Heap-Allocation Problem
    // instead inquire the core log
    vislib::sys::Log *corelog = nullptr;
    MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_CORELOG, MMC_TYPE_VOIDP, static_cast<void*>(&corelog)));
    if (corelog != nullptr) {
        vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::RedirectTarget(corelog, vislib::sys::Log::LEVEL_ALL));
        vislib::sys::Log::DefaultLog.EchoOfflineMessages(true);
    }

    void (MEGAMOLCORE_CALLBACK *echoFunc)(unsigned int level, const char* message)
#ifdef _WIN32
        = (parser->NoConColour())
        ? writeLogEchoToConsole
        : megamol::console::utility::windowsConsoleLogEcho;
#else
        = writeLogEchoToConsole;
#endif

    MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_LOGECHOFUNC, MMC_TYPE_VOIDP, function_cast<void*>(echoFunc)));
    if (parser->IsLogFileSpecified()) {
        MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_LOGFILE, MMC_TYPE_TSTR, parser->LogFile()));
    }
    if (parser->IsLogLevelSpecified()) {
        unsigned int level = parser->LogLevel();
        MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_LOGLEVEL, MMC_TYPE_UINT32, &level));
    }
    if (parser->IsLogEchoLevelSpecified()) {
        unsigned int level = parser->LogEchoLevel();
        MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_LOGECHOLEVEL, MMC_TYPE_UINT32, &level));
    }

    // Config file and config sets

    if (parser->IsConfigFileSpecified()) {
        MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_CFGFILE, MMC_TYPE_TSTR, parser->ConfigFile()));
    }

    vislib::TMultiSz cfgVals;
    cfgVals = parser->ConfigValues();
    ASSERT((cfgVals.Count() % 2) == 0);
    size_t cfgValsCount = cfgVals.Count() / 2;
    vislib::TString val;
    for (size_t i = 0; i < cfgValsCount; i++) {
        //::mmcSetConfigurationValue(hCore, MMC_CFGID_VARIABLE, cfgVals[i * 2], cfgVals[i * 2 + 1]);
        val.Append(cfgVals[i * 2]);
        val.Append(_T("\a"));
        val.Append(cfgVals[i * 2 + 1]);
        if (i != cfgValsCount - 1) {
            val.Append(_T("\b"));
        }
    }
    if (cfgValsCount > 0) {
        MMC_VERIFY_THROW(::mmcSetInitialisationValue(hCore, MMC_INITVAL_CFGOVERRIDE, MMC_TYPE_TSTR, val));
    }

    // Initialize core instance
    MMC_VERIFY_THROW(::mmcInitialiseCoreInstance(hCore));

    // window positions

    vislib::TMultiSz winPoss;
    winPoss = parser->WindowPositions();
    size_t possCnt = winPoss.Count();
    if (possCnt > 0) {
        for (size_t i = 0; i < possCnt - 1; i += 2) {
            vislib::TString winId = winPoss[i];
            vislib::TString winInf = winPoss[i + 1];
            vislib::TString cfgName(winId);
            cfgName += _T("-window");
            megamol::console::utility::WindowPlacement pw;
            pw.Parse(winInf);
            vislib::TString cfgVal(pw.ToString());
            if (!::mmcSetConfigurationValue(hCore, MMC_CFGID_VARIABLE, cfgName, cfgVal)) {
                vislib::sys::Log::DefaultLog.WriteWarn("Failed to reflect window settings for %s", vislib::StringA(winId).PeekBuffer());
            }
        }
    }

    // Further configurations

    if (parser->SetVSync()) {
        if (!::mmcSetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("vsync"), parser->SetVSyncOff() ? _T("off") : _T("on"))) {
            vislib::sys::Log::DefaultLog.WriteWarn("Failed to set vsync parameter");
        }
    }

    if (parser->ShowGUI() || parser->HideGUI()) {
        if (!::mmcSetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("consolegui"), parser->ShowGUI() ? _T("true") : _T("false"))) {
            vislib::sys::Log::DefaultLog.WriteWarn("Failed to set consolegui parameter");
        }
    }

    if (parser->UseKHRDebug()) {
        if (!::mmcSetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("useKHRdebug"), parser->UseKHRDebug() ? _T("true") : _T("false"))) {
            vislib::sys::Log::DefaultLog.WriteWarn("Failed to set KHR debug");
        }
    }

    // Quickstarts
    // warn on deprecated quickstart arguments
    vislib::SingleLinkedList<vislib::TString> quickstarts;
    parser->GetQuickstarts(quickstarts);
    if (parser->HasQuickstartRegistrations() || !quickstarts.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("Quickstarts are no longer supported. All corresponding command line arguments are ignored.");
    }

    megamol::console::utility::ParamFileManager::Instance().hCore = hCore;
    megamol::console::utility::ParamFileManager::Instance().filename = parser->ParameterFile();

#ifdef _WIN32
    megamol::console::utility::windowsConsoleWindowSetup(hCore);
#endif /* _WIN32 */
}

void processPendingActions(void) {
    while (::mmcHasPendingJobInstantiationRequests(hCore)) {
        if (!megamol::console::JobManager::Instance().InstantiatePendingJob(hCore)) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to instantiate the requested job.");
            vislib::sys::Log::DefaultLog.WriteError("Skipping remaining instantiation requests");
            break;
        }
    }
    while (::mmcHasPendingViewInstantiationRequests(hCore)) {
        if (!megamol::console::WindowManager::Instance().InstantiatePendingView(hCore)) { // <-- GLFW and OpenGL context/window start life when instantiating the first view
            vislib::sys::Log::DefaultLog.WriteError("Unable to instantiate the requested view.");
            vislib::sys::Log::DefaultLog.WriteError("Skipping remaining instantiation requests");
            break;
        }
    }
    ::mmcPerformGraphUpdates(hCore);
}

/**
 * Implements normal program operations
 *
 * @param parser Pointer to the parser object holding start-up arguments
 *
 * @return The return code value
 */
int runNormal(megamol::console::utility::CmdLineParser *& parser) {
#ifndef _WIN32
    oldSignl =
#endif /* !_WIN32 */
    signal(SIGINT, signalCtrlC);

    // Create an instance
    if (!createCoreInstance()) return -21;
    try { // initialise core
        setupCore(parser);

    } catch (vislib::Exception ex) {
        std::cerr << "Unable to initialise core instance: " << ex.GetMsgA() << " [" << ex.GetFile() << ":" << ex.GetLine() << "]" << std::endl;
        return -22;
    } catch (...) {
        std::cerr << "Unable to initialise core instance." << std::endl;
        return -22;
    }

    // prepare project files and instantiations
    vislib::SingleLinkedList<vislib::TString> projects;
    parser->GetProjectFiles(projects);
    int loadedProjects = 0;
    int loadedLuaProjects = 0;
    countProjects(projects, loadedProjects, loadedLuaProjects);
    if (loadedLuaProjects > 0 && loadedLuaProjects != loadedProjects) {
        vislib::sys::Log::DefaultLog.WriteError("You cannot mix loading legacy projects and lua projects!");
        return -66;
    }

    if (loadedLuaProjects == 0) {
    // try to create all requested instances
    // Remember the ids so we can predict view names before creating them
    // later.
        vislib::TMultiSz insts;
        insts = parser->Instantiations();
        //vislib::SingleLinkedList<vislib::TString> instanceNames;
        ASSERT((insts.Count() % 2) == 0);
        size_t instsCnt = insts.Count() / 2;
        for (size_t i = 0; i < instsCnt; i++) {
            ::mmcRequestInstance(hCore, insts[i * 2], insts[i * 2 + 1]);
            //instanceNames.Add(insts[i * 2 + 1]);
        }
        // If no instatiations have been requested through the command line and the
        // 'loadall' flag is set request all instatiations found in all
        // provided project files
        if ((instsCnt == 0) && (parser->LoadAll())) {
            ::mmcRequestAllInstances(hCore);
        }
        processPendingActions();
    }

    processPendingActions(); // <-- GLFW and OpenGL context/window start life here!

    // parameter value options
    std::map<vislib::TString, vislib::TString> paramValues;
    parser->GetParameterValueOptions(paramValues);
    for (auto& pv : paramValues) {
        ::megamol::console::CoreHandle hParam;
        if (!::mmcGetParameter(hCore, pv.first, hParam)) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to get handle for parameter \"%s\".\n", vislib::StringA(pv.first).PeekBuffer());
            continue;
        }
        ::mmcSetParameterValue(hParam, pv.second);
    }

    // parameter file options
    if (!megamol::console::utility::ParamFileManager::Instance().filename.IsEmpty()) {
        if (parser->InitParameterFile() || parser->InitOnlyParameterFile()) {
            megamol::console::utility::ParamFileManager::Instance().Save();
            if (parser->InitOnlyParameterFile()) return 0;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            megamol::console::utility::ParamFileManager::Instance().Load();
        }
    }

    // main loop
    bool winsAlive, jobsAlive;
    do {
        winsAlive = megamol::console::WindowManager::Instance().IsAlive();
        jobsAlive = megamol::console::JobManager::Instance().IsAlive();

        if (jobsAlive) {
            if (terminationRequest) {
                megamol::console::JobManager::Instance().Shutdown();
            }
            megamol::console::JobManager::Instance().Update();
        }
        if (winsAlive) {
            if (terminationRequest) {
                megamol::console::WindowManager::Instance().Shutdown();
            }
            megamol::console::WindowManager::Instance().Update();
        }
        processPendingActions();
    } while (winsAlive || jobsAlive);

#if 0


    // run the application!


        // Initialise views
#ifndef NOWINDOWPOSFIX
            int predictedX, predictedY, predictedWidth, predictedHeight;
            bool predictedNdFlag;

            // Predict the name of the next view. We expect the list to be
            // non-empty or mmcHasPendingViewInstantiationRequests should
            // not have returned true (this is somewhat dodgy, as it assumes
            // the core instantiates exactly those instances we requested in
            // exactly the order we requested them).
            vislib::TString viewName;
            if (instanceNames.Count() > 0) {
                viewName = instanceNames.First();
                instanceNames.RemoveFirst();
            } else {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "More views are "
                    "instantiated than were specified on the command line. "
                    "This may currently confuse the GPU affinity code.");
            }

            // Create a render context for this window. Not sure how best to
            // send a message to the window. For now, set the title to nullptr
            // (will be replaced later) which, for the WGLViewer, will trigger
            // the creation of the affinity context.
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "The GPU "
                "affinity will be determined based on the assumption that "
                "view \"%s\" is located at (%d, %d), size (%d, %d).",
                viewName.PeekBuffer(), predictedX, predictedY, predictedWidth,
                predictedHeight);
#endif

            if (::mmvSupportContextMenu(win->HWnd())) {
                ::mmvInstallContextMenu(win->HWnd());

                ::mmvInstallContextMenuCommandA(win->HWnd(), "Save State to 'YY-MM-dd_hh-mm-ss.mmprj'", 5);
                vislib::StringA saveAsMenuStr = "Save State to '";
                saveAsMenuStr.Append(mainProjectFile);
                saveAsMenuStr.Append("' (old)");
                ::mmvInstallContextMenuCommandA(win->HWnd(), saveAsMenuStr, 6);
            }

            }
        }
    }

#endif

    return 0;
}

void countProjects(vislib::SingleLinkedList<vislib::TString> &projects, int &loadedProjects, int &loadedLuaProjects)
{
    vislib::SingleLinkedList<vislib::TString>::Iterator projectIter = projects.GetIterator();
    while (projectIter.HasNext()) {
        const vislib::TString& project = projectIter.Next();
        // HAZARD: Legacy Projects vs. new Projects
        ::mmcLoadProject(hCore, project);
        loadedProjects++;
        if (project.EndsWith(".lua") || project.EndsWith(".png")) {
            loadedLuaProjects++;
            processPendingActions();
        }
    }
}

/**
 * Signal handler for Ctrl+C to change from abort to clean program shutdown.
 *
 * @param The signal to be handled.
 */
void signalCtrlC(int) {
    std::cerr << "SIGINT received. Program termination requested." << std::endl;
    terminationRequest = true;
#ifndef _WIN32
    signal(SIGINT, oldSignl);
#endif /* !_WIN32 */
}

/**
 * Writes a log message echo to the console.
 *
 * @param level The level of the log message.
 * @param message The text of the log message.
 */
void MEGAMOLCORE_CALLBACK writeLogEchoToConsole(unsigned int level, const char* message) {
    std::cout << std::setw(4) << level << "|" << message;
}

}
