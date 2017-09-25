/*
 * CmdLineParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_CMDLINEPARSER_H_INCLUDED
#define MEGAMOLCON_CMDLINEPARSER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/sys/CmdLineParser.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/MultiSz.h"
#include <map>


namespace megamol {
namespace console {
namespace utility {

    /**
     * Wrapper class for the command line parser
     */
    class CmdLineParser {
    public:

        /** ctor */
        CmdLineParser(void);

        /** dtor */
        ~CmdLineParser(void);

        /**
         * Parses the command line.
         *
         * @param argc The number of command line arguments
         * @param argv The command line argument strings.
         * @param includeFirstArgument Flag whether or not to include the 
         *                             first argument in the parsing.
         *
         * @return Zero or a positive value on success. If The value is above
         *         zero warnings are present. If the value is below zero errors
         *         are present.
         */
        int Parse(int argc, TCHAR **argv);

        /**
         * Prints all errors and warnings produced by the last parser run.
         *
         * @param output The output stream to print the information to.
         */
        void PrintErrorsAndWarnings(FILE* output);

        /**
         * Prints the parser online help
         */
        void PrintHelp(void);

        /**
         * Answer whether the help message was requested.
         *
         * @return 'true' if the help message was requested, 'false' otherwise.
         */
        inline bool WantHelp(void) const {
            return (this->help.GetFirstOccurrence() != NULL)
                || (this->fullHelp.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer whether the version information was requested.
         *
         * @return 'true' if the version information was requested, 'false' otherwise.
         */
        inline bool ShowVersionInfo(void) const {
            return (this->versionInfo.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer whether the startup logo should be hidden.
         *
         * @return 'true' if the startup logo should be hidden, 'false' otherwise.
         */
        inline bool HideLogo(void) const {
            return (this->noLogo.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer whether coloured text can be used in the console.
         *
         * @return 'true' if no console colours should be used,
         *         'false' otherwise.
         */
        inline bool NoConColour(void) const {
            return (this->noConColour.GetFirstOccurrence() != NULL);
        }

        /**
         * Flag whether to show the command line startup echo.
         *
         * @return 'true' if the command line should be echoed on startup,
         *         'false' otherwise.
         */
        bool EchoCmdLine(void) const;

        /**
         * Answers whether the config file is specified or not.
         *
         * @return 'true' if the config file is specified, 'false' otherwise.
         */
        inline bool IsConfigFileSpecified(void) const {
            return (this->configFile.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer the config file to load.
         *
         * @return The specified config file to load.
         */
        inline const TCHAR* ConfigFile(void) const {
            ParserArgument *arg = this->configFile.GetFirstOccurrence();
            return (arg == NULL) ? NULL : arg->GetValueString();
        }

        /**
        * Answers all names and values of config values that are to be overridden.
        *
        * @return All names and values of config values
        */
        vislib::TMultiSz ConfigValues(void) const;

        /**
         * Answers whether the log file is specified or not.
         *
         * @return 'true' if the log file is specified, 'false' otherwise.
         */
        inline bool IsLogFileSpecified(void) const {
            return (this->logFile.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer the log file to use.
         *
         * @return The specified log file to use.
         */
        inline const TCHAR* LogFile(void) const {
            ParserArgument *arg = this->logFile.GetFirstOccurrence();
            return (arg == NULL) ? NULL : arg->GetValueString();
        }

        /**
         * Answers whether the log level is specified or not.
         *
         * @return 'true' if the log level is specified, 'false' otherwise.
         */
        inline bool IsLogLevelSpecified(void) const {
            return (this->logLevel.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer the log level to use.
         *
         * @return The specified log level to use.
         */
        inline unsigned int LogLevel(void) const {
            return this->parseAsLogLevel(this->logLevel.GetFirstOccurrence());
        }

        /**
         * Answers whether the log echo level is specified or not.
         *
         * @return 'true' if the log echo level is specified, 
         *         'false' otherwise.
         */
        inline bool IsLogEchoLevelSpecified(void) const {
            return (this->logEchoLevel.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer the log echo level to use.
         *
         * @return The specified log echo level to use.
         */
        inline unsigned int LogEchoLevel(void) const {
            return this->parseAsLogLevel(this->logEchoLevel.GetFirstOccurrence());
        }

        /**
         * Answer whether the viewer module is forced to be loaded and started.
         *
         * @return 'true' if the viewer module is forced to be loaded and
         *         started, 'false' otherwise.
         */
        inline bool IsViewerForced(void) const {
            return (this->forceViewer.GetFirstOccurrence() != NULL);
        }

        /**
         * Answers all names of 'jobs' and 'views' to be instantiated.
         *
         * @return All names of 'jobs' and 'views' to be instantiated.
         */
        vislib::TMultiSz Instantiations(void) const;

        /**
         * Answer whether the command line file option is present.
         *
         * @return 'true' if the command line file option is present,
         *         'false' otherwise.
         */
        inline bool UseCmdLineFile(void) const {
            return (this->cmdLineFile.GetFirstOccurrence() != NULL);
        }

        /**
         * Answer the command line file to be used, or NULL if none is
         * specified.
         *
         * @return The command line file to be used.
         */
        inline const TCHAR* CmdLineFile(void) const {
            ParserArgument *arg = this->cmdLineFile.GetFirstOccurrence();
            return (arg == NULL) ? NULL : arg->GetValueString();
        }

        /**
         * Gets the parameter-value pairs which should be set.
         *
         * @param outParamValues The parameter-value pairs to be set.
         */
        void GetParameterValueOptions(std::map<vislib::TString, vislib::TString>& outParamValues) const;

        /**
         * Answers a list of file names of projects to be load.
         *
         * @param outFiles Receives the file names of projects to be load.
         */
        void GetProjectFiles(
            vislib::SingleLinkedList<vislib::TString>& outFiles) const;

        /**
         * Answer the parameter file to use, or an empty string in case no
         * parameter file was specified.
         *
         * @return The parameter file to use.
         */
        inline vislib::TString ParameterFile(void) const {
            ParserArgument *arg = this->paramFile.GetFirstOccurrence();
            return (arg == NULL) ? _T("") : arg->GetValueString();
        }

        /**
         * Answers whether or not to initialize the parameter file
         *
         * @return true if the parameter file should be initialized.
         */
        inline bool InitParameterFile(void) const {
            ParserArgument *arg1 = this->paramFile.GetFirstOccurrence();
            ParserArgument *arg2 = this->paramFileInit.GetFirstOccurrence();
            ParserArgument *arg3 = this->paramFileInitOnly.GetFirstOccurrence();
            return (arg1 != NULL) && ((arg2 != NULL) || (arg3 != NULL));
        }

        /**
         * Answers whether or not to initialize the parameter file
         *
         * @return true if the parameter file should be initialized.
         */
        inline bool InitOnlyParameterFile(void) const {
            ParserArgument *arg1 = this->paramFile.GetFirstOccurrence();
            ParserArgument *arg3 = this->paramFileInitOnly.GetFirstOccurrence();
            return (arg1 != NULL) && (arg3 != NULL);
        }

        /**
         * Answers the window placement commands.
         *
         * @return The window placement commands.
         */
        vislib::TMultiSz WindowPositions(void) const;

        /**
         * Answer if the set vsync option is present
         *
         * @return 'True' if the set vsync option is present
         */
        inline bool SetVSync(void) const {
            ParserArgument *arg = this->setVSync.GetFirstOccurrence();
            return (arg != NULL);
        }

        /**
         * Answers if the vsync should be turned off
         *
         * @return 'True' if the vsync should be turned off
         */
        inline bool SetVSyncOff(void) const {
            ParserArgument *arg = this->setVSync.GetFirstOccurrence();
            try {
                return ((arg != NULL) && !arg->GetValueBool());
            } catch(...) {
            }
            return false;
        }

        /**
         * Answer if the gui should be shown
         *
         * @return 'True' if the gui should be shown
         */
        inline bool ShowGUI(void) const {
            ParserArgument *arg = this->showGUI.GetFirstOccurrence();
            try {
                return ((arg != NULL) && arg->GetValueBool());
            } catch(...) {
            }
            return false;
        }

        /**
         * Answer if the gui should be hidden
         *
         * @return 'True' if the gui should be hidden
         */
        inline bool HideGUI(void) const {
            ParserArgument *arg = this->showGUI.GetFirstOccurrence();
            try {
                return ((arg != NULL) && !arg->GetValueBool());
            } catch(...) {
            }
            return false;
        }

        /**
        * Answer if the KHR debugging should be activated
        *
        * @return 'True' if the KHR should be used, false if not
        */
        inline bool UseKHRDebug(void) const {
            ParserArgument *arg = this->useKHR.GetFirstOccurrence();
            try {
                return ((arg != NULL) && arg->GetValueBool());
            }
            catch (...) {
            }
            return false;
        }

        /**
         * Answer if opengl quad buffers should be requested.
         *
         * @return true if opengl quad buffers should be requested.
         */
        inline bool RequestOpenGLQuadBuffer(void) const {
            return this->quadBuffer.GetFirstOccurrence() != NULL;
        }

        /**
         * Answers if any quickstarts have been specified
         *
         * @return True if quickstarts have been specified
         */
        inline bool HasQuickstarts(void) const {
            return this->quickstart.GetFirstOccurrence() != NULL;
        }

        /**
         * Gets the list of specified quickstarts
         *
         * @param outQuickstarts List of specified quickstarts
         */
        void GetQuickstarts(vislib::SingleLinkedList<vislib::TString>& outQuickstarts) const;

        /**
         * Answers if any quickstart registration commands have been specified
         *
         * @return True if quickstart registration commands have been specified
         */
        inline bool HasQuickstartRegistrations(void) const {
            return this->quickstartRegistry.GetFirstOccurrence() != NULL;
        }

        /**
         * Gets the list of specified quickstart registration commands
         *
         * @param outQuickstartRegs List of specified quickstart registration commands
         */
        void GetQuickstartRegistrations(vislib::SingleLinkedList<vislib::TString>& outQuickstartRegs) const;

        /**
         * Answer whether all possible instances should be loaded.
         *
         * @return 'True' if all possible instances should be loaded.
         */
        inline bool LoadAll(void) const {
            ParserArgument *arg = this->loadAll.GetFirstOccurrence();
            try {
                return (arg != NULL);
            } catch(...) {
            }
            return false;
        }

    private:

        /** the default value of the flag of the cmd line echo */
        static const bool cmdLineEchoDef;

        /** typedef of parser options */
        typedef vislib::sys::TCmdLineParser::Argument ParserArgument;

        /** typedef of parser options */
        typedef vislib::sys::TCmdLineParser::Option ParserOption;

        /** typedef of parser options */
        typedef vislib::sys::TCmdLineParser::Option::ValueDesc ParserValueDesc;

        /**
         * Parses a parser argument as log level.
         *
         * @param arg The argument to be parsed.
         *
         * @return The level parsed or zero in case of any error.
         */
        unsigned int parseAsLogLevel(const ParserArgument* arg) const;

        /** the vislib parser object */
        vislib::sys::TCmdLineParser parser;

        /** The help option */
        ParserOption help;

        /** The full help option */
        ParserOption fullHelp;

        /** The version information option */
        ParserOption versionInfo;

        /** The no logo parser option */
        ParserOption noLogo;

        /** The no console colour parser option */
        ParserOption noConColour;

        /** activates or deactives the command line echo */
        ParserOption echoCmdLine;

        /** specifies the configuration file to load */
        ParserOption configFile;

        /** overrides a variable from the configuration */
        ParserOption configValue;

        /** specifies a log file to use */
        ParserOption logFile;

        /** specifies a log level to use */
        ParserOption logLevel;

        /** specifies a log echo level to use */
        ParserOption logEchoLevel;

        /** forces the loading of the viewer library */
        ParserOption forceViewer;

        /** instancing of a 'job' or 'view' */
        ParserOption instJobView;

        /** redirected command line file option */
        ParserOption cmdLineFile;

        /** sets a parameter value */
        ParserOption paramValue;

        /** adds a project file to the currently running core instance */
        ParserOption projectFile;

        /** specifies the parameter file to use */
        ParserOption paramFile;

        /** specified a window position */
        ParserOption winPos;

        /**
         * if present the parameter file is written as soon as all instances
         * have been created.
         */
        ParserOption paramFileInit;

        /**
         * if present the parameter file is written as soon as all instances
         * have been created and terminates the application afterwards.
         */
        ParserOption paramFileInitOnly;

        /** Parser option to control VSync */
        ParserOption setVSync;

        /** Parser option to de-/activate the gui layer */
        ParserOption showGUI;

        /** Parser option to de-/activate KHR debugging */
        ParserOption useKHR;

        /** Flag to request quad-buffer support */
        ParserOption quadBuffer;

        /** Perform a data set quickstart */
        ParserOption quickstart;

        /** Registers data sets for quickstart */
        ParserOption quickstartRegistry;

        /** Enables hot fixes */
        ParserOption enableHotFix;

        /** Flag, that causes all instances specified in all project files to
         *  be loaded. Is overwritten by using the '-i' option. */
        ParserOption loadAll;

    };

} /* end namespace utility */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_CMDLINEPARSER_H_INCLUDED */
