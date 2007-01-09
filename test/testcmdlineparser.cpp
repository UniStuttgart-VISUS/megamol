/*
 * testcmdlineparser.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testcmdlineparser.h"
#include "testhelper.h"

#include <vislib/Console.h>

#include "vislib/CmdLineParser.h"
#include "vislib/CmdLineProvider.h"

typedef vislib::sys::CmdLineParserA CLParser;
typedef vislib::sys::CmdLineParserA::Option CLPOption;

/*
 * Test of the command line parser.
 *
 * TODO: EXTEND the test to all interessting effects.
 */
void TestCmdLineParser(void) {

    {
        vislib::sys::CmdLineProviderA problemLine("\"Horst's stupid Programm\"");
        AssertEqual("problemLine.ArgC() == 1", problemLine.ArgC(), 1);
        AssertNotEqual<char**>("problemLine.ArgV() != NULL", problemLine.ArgV(), NULL);
        AssertEqual("problemLine.ArgV()[0] == Horst's stupid Programm", problemLine.ArgV()[0], "Horst's stupid Programm");
    }

    {
        vislib::sys::CmdLineProviderW testLine1(L"This is a \"Smart \"\"Stuff\"\"-Test\" of the cmdline");
        if (AssertEqual("testLine1.ArgC() == 7", testLine1.ArgC(), 7)) {
            AssertNotEqual<wchar_t**>("testLine1.ArgV() != NULL", testLine1.ArgV(), NULL);
            AssertEqual("testLine1.ArgV()[0] == This", testLine1.ArgV()[0], L"This");
            AssertEqual("testLine1.ArgV()[1] == is", testLine1.ArgV()[1], L"is");
            AssertEqual("testLine1.ArgV()[2] == a", testLine1.ArgV()[2], L"a");
            AssertEqual("testLine1.ArgV()[3] == Smart \"Stuff\"-Test", testLine1.ArgV()[3], L"Smart \"Stuff\"-Test");
            AssertEqual("testLine1.ArgV()[4] == of", testLine1.ArgV()[4], L"of");
            AssertEqual("testLine1.ArgV()[5] == the", testLine1.ArgV()[5], L"the");
            AssertEqual("testLine1.ArgV()[6] == cmdline", testLine1.ArgV()[6], L"cmdline");
        }

        vislib::sys::CmdLineProviderW testLine2( testLine1.SingleStringCommandLine(false) ); // rebuild cmd line and resplit
        if (AssertEqual("testLine2.ArgC() == 7", testLine2.ArgC(), 7)) {
            AssertNotEqual<wchar_t**>("testLine2.ArgV() != NULL", testLine2.ArgV(), NULL);
            AssertEqual("testLine2.ArgV()[0] == This", testLine2.ArgV()[0], L"This");
            AssertEqual("testLine2.ArgV()[1] == is", testLine2.ArgV()[1], L"is");
            AssertEqual("testLine2.ArgV()[2] == a", testLine2.ArgV()[2], L"a");
            AssertEqual("testLine2.ArgV()[3] == Smart \"Stuff\"-Test", testLine2.ArgV()[3], L"Smart \"Stuff\"-Test");
            AssertEqual("testLine2.ArgV()[4] == of", testLine2.ArgV()[4], L"of");
            AssertEqual("testLine2.ArgV()[5] == the", testLine2.ArgV()[5], L"the");
            AssertEqual("testLine2.ArgV()[6] == cmdline", testLine2.ArgV()[6], L"cmdline");
        }
    }

    vislib::StringA cmdName = vislib::sys::CmdLineProviderA::GetModuleName();
    vislib::sys::CmdLineProviderA cmdLine(cmdName, 
        "Horst 1.2 -h '' -hugo \"321 \"\"Heinz\" --tEst 4.2 Helmut -Help 11 -ht 4.1");

    AssertEqual("cmdLine.ArgC() == 14", cmdLine.ArgC(), 14);
    AssertNotEqual<char**>("cmdLine.ArgV() != NULL", cmdLine.ArgV(), NULL);
    AssertEqual("cmdLine.ArgV()[0] == GetModuleName", cmdLine.ArgV()[0], cmdName);
    AssertEqual("cmdLine.ArgV()[1] == Horst", cmdLine.ArgV()[1], "Horst");
    AssertEqual("cmdLine.ArgV()[2] == 1.2", cmdLine.ArgV()[2], "1.2");
    AssertEqual("cmdLine.ArgV()[3] == -h", cmdLine.ArgV()[3], "-h");
    AssertEqual("cmdLine.ArgV()[4] == ''", cmdLine.ArgV()[4], "''");
    AssertEqual("cmdLine.ArgV()[5] == -hugo", cmdLine.ArgV()[5], "-hugo");
    AssertEqual("cmdLine.ArgV()[6] == 321 \"Heinz", cmdLine.ArgV()[6], "321 \"Heinz");
    AssertEqual("cmdLine.ArgV()[7] == --tEst", cmdLine.ArgV()[7], "--tEst");
    AssertEqual("cmdLine.ArgV()[8] == 4.2", cmdLine.ArgV()[8], "4.2");
    AssertEqual("cmdLine.ArgV()[9] == Helmut", cmdLine.ArgV()[9], "Helmut");
    AssertEqual("cmdLine.ArgV()[10] == -Help", cmdLine.ArgV()[10], "-Help");
    AssertEqual("cmdLine.ArgV()[11] == 11", cmdLine.ArgV()[11], "11");
    AssertEqual("cmdLine.ArgV()[12] == -ht", cmdLine.ArgV()[10], "-ht");
    AssertEqual("cmdLine.ArgV()[13] == 4.1", cmdLine.ArgV()[11], "4.1");

    CLParser parser;
    CLPOption helpOption('h', "help");
    CLPOption testOption('t', "Test", "Just a fucking test option.", vislib::sys::CmdLineParserA::Option::DOUBLE_VALUE);

    AssertFalse("helpOption.IsValueValid() == false", helpOption.IsValueValid());
    AssertEqual("helpOption.GetValueType() == NO_VALUE", helpOption.GetValueType(), CLPOption::NO_VALUE);
    AssertFalse("testOption.IsValueValid() == false", testOption.IsValueValid());
    AssertEqual("testOption.GetValueType() == DOUBLE_VALUE", testOption.GetValueType(), CLPOption::DOUBLE_VALUE);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) == NULL", parser.FindOption("HeLp"), NULL);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption('t'), NULL);

    parser.AddOption(&helpOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption("HeLp"), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption('t'), NULL);

    parser.AddOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption("HeLp"), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(H) == NULL", parser.FindOption('H'), NULL);
    AssertEqual<const CLPOption*>("parser.FindOption(h) != NULL", parser.FindOption('h'), &helpOption);
    AssertNotEqual<const CLPOption*>("parser.FindOption(t) != NULL", parser.FindOption('t'), NULL);

    AssertException("Adding option a second time; Expecting IllegalParamException", parser.AddOption(&helpOption), vislib::IllegalParamException);

    parser.RemoveOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption("HeLp"), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption('t'), NULL);

    parser.AddOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption("HeLp"), &helpOption);
    AssertNotEqual<const CLPOption*>("parser.FindOption(t) != NULL", parser.FindOption('t'), NULL);

    AssertEqual("parser.Parse == 0", parser.Parse(cmdLine.ArgC(), cmdLine.ArgV()), 0);

    /*if (!AssertFalse("No Warnings", parser.GetWarnings().HasNext()))*/ { // TODO: Check!
        CLParser::WarningIterator wii = parser.GetWarnings();
        if (wii.HasNext()) {
            printf("Warnings:\n");
        }

        while (wii.HasNext()) {
            CLParser::Warning &w = wii.Next();
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::YELLOW);
            printf("Warning %d [Arg %u]: ", int(w.GetWarnCode()), w.GetArgument());
            vislib::sys::Console::RestoreDefaultColors();
            printf("%s\n", CLParser::Warning::GetWarningString(w.GetWarnCode()));
        }
    }

    /*if (!AssertFalse("No Errors", parser.GetErrors().HasNext()))*/ { // TODO: Check!
        CLParser::ErrorIterator err = parser.GetErrors();
        if (err.HasNext()) {
            printf("Errors:\n");
        }

        while (err.HasNext()) {
            CLParser::Error &e = err.Next();
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
            printf("Error %d [Arg %u]: ", int(e.GetErrorCode()), e.GetArgument());
            vislib::sys::Console::RestoreDefaultColors();
            printf("%s\n", e.GetErrorString(e.GetErrorCode()));
        }
    }

}
