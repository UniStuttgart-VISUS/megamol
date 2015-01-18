/*
 * clusterTest.cpp  12.4.2008
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten. 
 */

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <tchar.h>
#endif /* _WIN32 */

#include <iostream>

#include "vislib/CmdLineParser.h"
#include "vislib/String.h"
#include "vislib/Trace.h"

#include "GlutClient.h"
#include "GlutServer.h"
#include "PlainClient.h"
#include "PlainServer.h"
#include "DiscoveryTestApp.h"


void usage(const TCHAR *app, vislib::sys::TCmdLineParser& parser) {
    std::cerr << "Usage: " << app << " [options]" << std::endl;
    std::cerr << "Options are:" << std::endl << std::endl;

    vislib::sys::TCmdLineParser::OptionDescIterator it 
            = parser.OptionDescriptions(true);
    while (it.HasNext()) {
        std::cerr << it.Next() << std::endl;
    }
}


/**
 * Entry point of the application.
 *
 * @param argc An integer that contains the count of arguments that follow in 
 *             'argv'. The 'argc' parameter is always greater than or equal 
 *             to 1.
 * @param argv An array of null-terminated strings representing command-line 
 *             arguments entered by the user of the program. By convention, 
 *             argv[0] is the command with which the program is invoked, argv[1]
 *             is the first command-line argument, and so on.
 *
 * @return Application-specific return value.
 */
#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else /* _WIN32 */
int main(int argc, char **argv) {
#endif /* _WIN32 */
    typedef vislib::sys::TCmdLineProvider CmdLine;
    typedef vislib::sys::TCmdLineParser Parser;
    typedef Parser::Argument Argument;
    typedef Parser::Option Option;
    typedef Option::ValueDesc ValueDesc;

    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL_VERBOSE);

    Parser parser;
    Option optTest(_T("test"),
        _T("Specifies the test to run."),
        Option::FLAG_UNIQUE | Option::FLAG_REQUIRED,
        ValueDesc::ValueList(Option::STRING_VALUE, _T("name"), 
        _T("The name of the test mode to run. This can be one of ")
        _T("\"plainserver\", \"plainclient\", ")
        _T("\"glutserver\", \"glutclient\", ")
        _T("\"discovery\".")));
    parser.AddOption(&optTest);

    int parseResult = parser.Parse(argc, argv);
    if (parseResult < 0) {
        ::usage(argv[0], parser);
        return parseResult;
    }

    CmdLine cmdLine(argc, argv);
    Argument *arg = NULL;
    if ((arg = optTest.GetFirstOccurrence()) != NULL) {
        try {
            vislib::TString val = arg->GetValueString();
            if (val.Equals(_T("plainserver"), false)) {
                PlainServer::GetInstance().Initialise(cmdLine);
                return PlainServer::GetInstance().Run();
            } else if (val.Equals(_T("plainclient"), false)) {
                PlainClient::GetInstance().Initialise(cmdLine);
                return PlainClient::GetInstance().Run();
#if defined(VISLIB_CLUSTER_WITH_OPENGL) && (VISLIB_CLUSTER_WITH_OPENGL != 0)
            } else if (val.Equals(_T("glutserver"), false)) {
                GlutServer::GetInstance().Initialise(cmdLine);
                return GlutServer::GetInstance().Run();
            } else if (val.Equals(_T("glutclient"), false)) {
                GlutClient::GetInstance().Initialise(cmdLine);
                return GlutClient::GetInstance().Run();
#endif /*defined(VISLIB_CLUSTER_WITH_OPENGL) && ... */
            } else if (val.Equals(_T("discovery"), false)) {
                DiscoveryTestApp::GetInstance().Initialise(cmdLine);
                return DiscoveryTestApp::GetInstance().Run();
            }
        } catch (vislib::Exception& e) {
            std::cerr << e.GetMsgA() << " @ " << e.GetFile() << ":" 
                << e.GetLine() << std::endl;
            return -1;
        } catch (...) {
        }
    }
    /* Illegal argument if here. */

    ::usage(argv[0], parser);
    return -1;
}
