/*
 * Console.cpp
 *
 * Copyright (C) 2008-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "CmdLineParser.h"
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/sys/Log.h"
#include "vislib/StringConverter.h"
#include "AboutInfo.h"
#include "utility/HotFixes.h"


/*
 * megamol::console::utility::CmdLineParser::cmdLineEchoDef
 */
const bool megamol::console::utility::CmdLineParser::cmdLineEchoDef = true;


/*
 * megamol::console::utility::CmdLineParser::CmdLineParser
 */
megamol::console::utility::CmdLineParser::CmdLineParser(void) 
    : parser(),
    help(_T('h'), _T("help"), _T("Prints a help description text."), ParserOption::FLAG_EXCLUSIVE),
    fullHelp(0, _T("fullhelp"), _T("Prints an extended help description text."), ParserOption::FLAG_EXCLUSIVE),
    versionInfo(0, _T("version"), _T("Prints version information about the application and the loaded core module.")),
    noLogo(0, _T("nologo"), _T("Suppresses the startup logo banner.")),
    noConColour(0, _T("noconcol"), _T("Deactivates the use of coloured text in the console window.")),
    echoCmdLine(0, _T("cmdlineecho"), _T("Activates or deactivates the command line startup echo."), ParserOption::FLAG_NULL, 
        ParserValueDesc::ValueList(ParserOption::BOOL_VALUE, _T("Activate"), 
        _T("'True' or 'False' to activate or deactivate the command line startup echo."))),
    configFile(0, _T("configfile"), _T("Specifies the configuration file to load."), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("path"), _T("The configuration file to load."))),
    configValue(_T('o'), _T("configval"), _T("Overrides a value (potentially set) in the config file."), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("cfgvar"), _T("the name of the config value"))
        ->Add(ParserOption::STRING_VALUE, _T("cfgval"), _T("the value to set"))),
    logFile(0, _T("logfile"), _T("Specifies the log file."), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("path"), _T("The path to the log file to use."))),
    logLevel(0, _T("loglevel"), _T("Specifies the log level."), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::INT_OR_STRING_VALUE, _T("level"), _T("The log level to use (Positive number or 'all')."))),
    logEchoLevel(0, _T("logecholevel"), _T("Specifies the log echo level."), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::INT_OR_STRING_VALUE, _T("level"), _T("The log level to use (Positive number or 'all')."))),
    forceViewer(0, _T("viewer"), _T("Forces the loading and starting of the viewer module"), ParserOption::FLAG_UNIQUE),
    instJobView(_T('i'), _T("instantiate"), _T("Instantiates a \"job\" or \"view\" from the loaded project (only works for legacy XML projects!)"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("name"), _T("The name of the \"job\" or \"view\" to be instantiated"))
        ->Add(ParserOption::STRING_VALUE, _T("id"), _T("The identifier name to be set for the instance"))),
    cmdLineFile(0, _T("cmdlinefile"), _T("Reads the first text line form a file and uses this line as command line"), ParserOption::FLAG_EXCLUSIVE,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("file"), _T("The text file to read the command line from"))),
    paramValue(_T('v'), _T("paramvalue"), _T("Sets the value of a parameter slot"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("name"), _T("The name of the parameter slot to set"))
        ->Add(ParserOption::STRING_VALUE, _T("value"), _T("The value to be set for the specified parameter"))),
    projectFile(_T('p'), _T("project"), _T("Loads the specified project file"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("file"), _T("The path of the project file to load"))),
    paramFile(0, _T("paramfile"), _T("Specifies the parameter file to use"), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("file"), _T("The path of the parameter file to use"))),
    winPos(_T('w'), _T("winpos"), _T("Specifies a window position"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("id"), _T("The view instance id specifying the window to place"))
        ->Add(ParserOption::STRING_VALUE, _T("placement"), _T("Specifies the window placement (Syntax: \"{ F[<num>] | [X<num>Y<num>][W<num>H<num>][nd] }\")"))),
    paramFileInit(0, _T("paramfileinit"), _T("if present the parameter file is written as soon as all instances have been created. (Has no effect if no parameter file is specified.)")),
    paramFileInitOnly(0, _T("paramfileinitonly"), _T("if present the parameter file is written as soon as all instances have been created and terminates the application afterwards. (Has no effect if no parameter file is specified.)")),
    setVSync(0, _T("vsync"), _T("(De-)Activates the vsync for all windows."), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::BOOL_VALUE, _T("switch"), _T("'True' forces vsync enable, 'False' forces vsync disable"))),
    showGUI(0, _T("gui"), _T("Option to de-/activate the gui layer"), ParserOption::FLAG_UNIQUE,
        ParserValueDesc::ValueList(ParserOption::BOOL_VALUE, _T("switch"), _T("'True' activates the gui, 'False' deactivates the gui"))),
    useKHR(0, _T("useKHRdebug"), _T("Option to de-/activate the KHR debugger"), ParserOption::FLAG_UNIQUE,
    ParserValueDesc::ValueList(ParserOption::BOOL_VALUE, _T("switch"), _T("'True' activates KHR debugging, 'False' deactivates KHR debugging"))),
    quadBuffer(0, _T("quadbuffer"), _T("Enables OpenGL Quad-Buffer support, if the viewer is started")),
    quickstart(_T('q'), _T("quickstart"), _T("(DEPRECATED!) Performs a quickstart for the specified data set"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("file"), _T("Path to the data file to quickstart"))),
    quickstartRegistry(0, _T("quickstartreg"), _T("(DEPRECATED!) Registers data file types for quickstart is supported by the OS"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("options"), _T("The quickstart registration options"))),
    enableHotFix(0, _T("hotfix"), _T("Enables a hot fix"), ParserOption::FLAG_NULL,
        ParserValueDesc::ValueList(ParserOption::STRING_VALUE, _T("name"), _T("The name of the hot fix to enable"))),
    loadAll(_T('a'), _T("loadall"), _T("Causes all instances specified in all provided project files to be loaded (has no effect if '--instantiate' or '-i' is used)."))
{

    this->parser.AddOption(&this->help);
    this->parser.AddOption(&this->fullHelp);
    this->parser.AddOption(&this->versionInfo);

    this->parser.AddOption(&this->noLogo);
    this->parser.AddOption(&this->noConColour);

    this->parser.AddOption(&this->echoCmdLine);
    this->parser.AddOption(&this->cmdLineFile);

    this->parser.AddOption(&this->configFile);
    this->parser.AddOption(&this->configValue);

    this->parser.AddOption(&this->logFile);
    this->parser.AddOption(&this->logLevel);
    this->parser.AddOption(&this->logEchoLevel);

    this->parser.AddOption(&this->forceViewer);
    this->parser.AddOption(&this->setVSync);
    this->parser.AddOption(&this->showGUI);
    this->parser.AddOption(&this->useKHR);
    this->parser.AddOption(&this->quadBuffer);

    this->parser.AddOption(&this->projectFile);
    this->parser.AddOption(&this->instJobView);
    this->parser.AddOption(&this->paramValue);
    this->parser.AddOption(&this->paramFile);
    this->parser.AddOption(&this->paramFileInit);
    this->parser.AddOption(&this->paramFileInitOnly);
    this->parser.AddOption(&this->winPos);

    this->parser.AddOption(&this->quickstart);
    this->parser.AddOption(&this->quickstartRegistry);

    this->parser.AddOption(&this->enableHotFix);
    this->parser.AddOption(&this->loadAll);

}


/*
 * megamol::console::utility::CmdLineParser::~CmdLineParser
 */
megamol::console::utility::CmdLineParser::~CmdLineParser(void) {
}


/*
 * megamol::console::utility::CmdLineParser::Parse
 */
int megamol::console::utility::CmdLineParser::Parse(int argc, TCHAR **argv) {
    int retval = this->parser.Parse(argc, argv, false);
    if (retval >= 0) {
        HotFixes& fs = const_cast<HotFixes&>(HotFixes::Instance());
        fs.Clear();

        for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
            ParserArgument *arg = this->parser.GetArgument(i);
            if (((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME) && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) || (arg->GetOption() != &this->enableHotFix)) {
                continue;
            }
            arg = this->parser.NextArgument(arg);
            i++;
            fs.EnableHotFix(T2A(arg->GetValueString()));
        }
    }
    return retval;
}


/*
 * megamol::console::utility::CmdLineParser::PrintErrorsAndWarnings
 */
void megamol::console::utility::CmdLineParser::PrintErrorsAndWarnings(FILE* output) {
    vislib::sys::TCmdLineParser::ErrorIterator errors = this->parser.GetErrors();
    vislib::sys::TCmdLineParser::WarningIterator warnings = this->parser.GetWarnings();
    bool requiredOptMissing = false;

    if (errors.HasNext()) {
        fprintf(output, "Parser Errors:\n");
        while (errors.HasNext()) {
            vislib::sys::TCmdLineParser::Error &e = errors.Next();
            fprintf(output, "[Arg.%3u] %s\n", e.GetArgument(),
                vislib::sys::TCmdLineParser::Error::GetErrorString(e.GetErrorCode()));
            if (e.GetErrorCode() == vislib::sys::TCmdLineParser::Error::MISSING_REQUIRED_OPTIONS) {
                requiredOptMissing = true;
            }
        }
        fprintf(output, "\n");
    }

    if (warnings.HasNext()) {
        fprintf(output, "Parser Warnings:\n");
        while (warnings.HasNext()) {
            vislib::sys::TCmdLineParser::Warning &w = warnings.Next();
            fprintf(output, "[Arg.%3u] %s\n", w.GetArgument(),
                vislib::sys::TCmdLineParser::Warning::GetWarningString(w.GetWarnCode()));
        }
        fprintf(output, "\n");
    }

    if (requiredOptMissing) {
        fprintf(output, "Missing required options:\n");
        vislib::sys::TCmdLineParser::OptionPtrList list 
            = this->parser.ListMissingRequiredOptions();
        vislib::sys::TCmdLineParser::OptionPtrList::Iterator i = list.GetIterator();
        while (i.HasNext()) {
            ParserOption* o = i.Next();
            vislib::TString name = o->GetLongName();
            if (name.IsEmpty()) {
                name = vislib::TString(o->GetShortName(), 1);
            }

            fprintf(output, "%s\n", vislib::StringA(T2A(name)).PeekBuffer());
        }
    }

}


/*
 * megamol::console::utility::CmdLineParser::PrintHelp
 */
void megamol::console::utility::CmdLineParser::PrintHelp(void) {
    bool fullHelp = (this->fullHelp.GetFirstOccurrence() != NULL);
    vislib::sys::TCmdLineParser::OptionDescIterator helptext 
        = this->parser.OptionDescriptions(fullHelp);

    printf("\nVersion:\n");
    megamol::console::utility::AboutInfo::PrintVersionInfo();

    printf("\n");
    while (helptext.HasNext()) {
        printf("%s\n", vislib::StringA(T2A(helptext.Next())).PeekBuffer());
    }

    printf("\nFor additional information consult the manual or the webpage:\n");
    printf("    http://megamol.org\n\n");
}


/*
 * megamol::console::utility::CmdLineParser::EchoCmdLine
 */
bool megamol::console::utility::CmdLineParser::EchoCmdLine(void) const {
    ParserArgument *arg = this->echoCmdLine.GetFirstOccurrence();
    try {
        if (arg != NULL) {
            return arg->GetValueBool();
        }
    } catch(...) {
    }
    return cmdLineEchoDef;
}


/*
* megamol::console::utility::CmdLineParser::ConfigValues
*/
vislib::TMultiSz
megamol::console::utility::CmdLineParser::ConfigValues(void) const {
    vislib::TMultiSz retval;
    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if ((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) {
            continue;
        }
        if (arg->GetOption() == &this->configValue) {
            arg = this->parser.NextArgument(arg);
            retval.Append(arg->GetValueString());   // name
            arg = this->parser.NextArgument(arg);
            retval.Append(arg->GetValueString());   // value
        }
    }
    return retval;
}


/*
 * megamol::console::utility::CmdLineParser::Instantiations
 */
vislib::TMultiSz
megamol::console::utility::CmdLineParser::Instantiations(void) const {
    vislib::TMultiSz retval;
    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if ((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) {
            continue;
        }
        if (arg->GetOption() == &this->instJobView) {
            arg = this->parser.NextArgument(arg);
            retval.Append(arg->GetValueString());   // name
            arg = this->parser.NextArgument(arg);
            retval.Append(arg->GetValueString());   // id
        }
    }
    return retval;
}


/*
 * megamol::console::utility::CmdLineParser::GetParameterValueOptions
 */
void megamol::console::utility::CmdLineParser::GetParameterValueOptions(std::map<vislib::TString, vislib::TString>& outParamValues) const {
    outParamValues.clear();
    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if ((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) {
            continue;
        }
        if (arg->GetOption() == &this->paramValue) {
            arg = this->parser.NextArgument(arg);
            vislib::TString name = arg->GetValueString(); // name
            arg = this->parser.NextArgument(arg);
            outParamValues[name] = arg->GetValueString(); // value
        }
    }
}


/*
 * megamol::console::utility::CmdLineParser::GetProjectFiles
 */
void megamol::console::utility::CmdLineParser::GetProjectFiles(
        vislib::SingleLinkedList<vislib::TString>& outFiles) const {
    outFiles.Clear();
    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if ((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) {
            continue;
        }
        if (arg->GetOption() == &this->projectFile) {
            outFiles.Add(arg->GetValueString());
        }
    }
}


/*
 * megamol::console::utility::CmdLineParser::WindowPositions
 */
vislib::TMultiSz
megamol::console::utility::CmdLineParser::WindowPositions(void) const {
    vislib::TMultiSz retval;
    retval.Clear(); // paranoia

    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if ((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) {
            continue;
        }
        if (arg->GetOption() == &this->winPos) {
            arg = this->parser.NextArgument(arg);
            vislib::TString id = arg->GetValueString();
            arg = this->parser.NextArgument(arg);
            vislib::TString val = arg->GetValueString();

            if (id.IsEmpty() || val.IsEmpty()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Parser option \"--winpos\" with empty \"id\" or \"value\" ignored");
            } else {
                retval.Append(id);
                retval.Append(val);
            }
        }
    }

    ASSERT((retval.Count() % 2) == 0);

    return retval;
}


/*
 * megamol::console::utility::CmdLineParser::GetQuickstarts
 */
void megamol::console::utility::CmdLineParser::GetQuickstarts(
        vislib::SingleLinkedList<vislib::TString>& outQuickstarts) const {
    outQuickstarts.Clear();

    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if (((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) 
            || (arg->GetOption() != &this->quickstart)) {
            continue;
        }
        arg = this->parser.NextArgument(arg);
        i++;
        outQuickstarts.Add(arg->GetValueString());
    }

}


/*
 * megamol::console::utility::CmdLineParser::GetQuickstartRegistrations
 */
void megamol::console::utility::CmdLineParser::GetQuickstartRegistrations(
        vislib::SingleLinkedList<vislib::TString>& outQuickstartRegs) const {
    outQuickstartRegs.Clear();

    for (unsigned int i = 0; i < this->parser.ArgumentCount(); i++) {
        ParserArgument *arg = this->parser.GetArgument(i);
        if (((arg->GetType() != ParserArgument::TYPE_OPTION_LONGNAME)
            && (arg->GetType() != ParserArgument::TYPE_OPTION_SHORTNAMES)) 
            || (arg->GetOption() != &this->quickstartRegistry)) {
            continue;
        }
        arg = this->parser.NextArgument(arg);
        i++;
        outQuickstartRegs.Add(arg->GetValueString());
    }
}


/*
 * megamol::console::utility::CmdLineParser::parseAsLogLevel
 */
unsigned int megamol::console::utility::CmdLineParser::parseAsLogLevel(
        const megamol::console::utility::CmdLineParser::ParserArgument* arg) 
        const {
    if (arg == NULL) {
        return 0;
    }
    try {
        vislib::TString str(arg->GetValueString());
        if (str.Equals(_T("error"), false)) {
            return vislib::sys::Log::LEVEL_ERROR;
        } else if (str.Equals(_T("warn"), false)) {
            return vislib::sys::Log::LEVEL_WARN;
        } else if (str.Equals(_T("warning"), false)) {
            return vislib::sys::Log::LEVEL_WARN;
        } else if (str.Equals(_T("info"), false)) {
            return vislib::sys::Log::LEVEL_INFO;
        } else if (str.Equals(_T("none"), false)) {
            return vislib::sys::Log::LEVEL_NONE;
        } else if (str.Equals(_T("null"), false)) {
            return vislib::sys::Log::LEVEL_NONE;
        } else if (str.Equals(_T("zero"), false)) {
            return vislib::sys::Log::LEVEL_NONE;
        } else if (str.Equals(_T("all"), false)) {
            return vislib::sys::Log::LEVEL_ALL;
        }
    } catch(...) {
    }
    try {
        int i = arg->GetValueInt();
        if (i < 0) {
            i = 0;
        }
        return static_cast<unsigned int>(i);
    } catch(...) {
    }

    return 0;
}
