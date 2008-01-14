/*
 * testcmdlineparser.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testcmdlineparser.h"
#include "testhelper.h"

#include <stdio.h>
#include <vislib/Console.h>

#include "vislib/CmdLineParser.h"
#include "vislib/CmdLineProvider.h"

// #define USE_UNICODE_PARSER

#ifdef USE_UNICODE_PARSER
typedef vislib::sys::CmdLineParserW CLParser;
typedef vislib::sys::CmdLineProviderW CLProvider;
#define CLPS(A) L ## A
typedef vislib::StringW CLPString;
#define CLPChar wchar_t

#else
typedef vislib::sys::CmdLineParserA CLParser;
typedef vislib::sys::CmdLineProviderA CLProvider;
#define CLPS(A) A
typedef vislib::StringA CLPString;
#define CLPChar char

#endif

typedef CLParser::Option CLPOption;
typedef CLParser::Argument CLPArgument;
typedef CLParser::Error CLPError;
typedef CLParser::Warning CLPWarning;


/*
 * Test of the command line parser.
 */
void TestCmdLineParser1(void) {

    { // subtest 1
        vislib::sys::CmdLineProviderA problemLine("\"Horst's stupid Programm\"");
        AssertEqual("problemLine.ArgC() == 1", problemLine.ArgC(), 1);
        AssertNotEqual<char**>("problemLine.ArgV() != NULL", problemLine.ArgV(), NULL);
        AssertEqual("problemLine.ArgV()[0] == Horst's stupid Programm", problemLine.ArgV()[0], "Horst's stupid Programm");
    } // subtest 1

    { // subtest 2
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

        vislib::sys::CmdLineProviderW testLine2( testLine1.SingleStringCommandLine(true) ); // rebuild cmd line and resplit
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
    } // subtest 2

    CLPString cmdName 
#ifdef _WIN32
        = CLProvider::GetModuleName();
#else // _WIN32
        = CLPS("vislibtest");
#endif // _WIN32
    CLProvider cmdLine(cmdName, 
        CLPS("Horst 1.2 -h \"\" -hugo \"321 \"\"Heinz\" --tEst 4.2 Helmut -Help 11 -ht 4.1"));

    AssertEqual("cmdLine.ArgC() == 14", cmdLine.ArgC(), 14);
    AssertNotEqual<CLPChar**>("cmdLine.ArgV() != NULL", cmdLine.ArgV(), NULL);
    AssertEqual("cmdLine.ArgV()[0] == GetModuleName", cmdLine.ArgV()[0], cmdName);
    AssertEqual("cmdLine.ArgV()[1] == Horst", cmdLine.ArgV()[1], CLPS("Horst"));
    AssertEqual("cmdLine.ArgV()[2] == 1.2", cmdLine.ArgV()[2], CLPS("1.2"));
    AssertEqual("cmdLine.ArgV()[3] == -h", cmdLine.ArgV()[3], CLPS("-h"));
    AssertEqual("cmdLine.ArgV()[4] == <empty String>", cmdLine.ArgV()[4], CLPS(""));
    AssertEqual("cmdLine.ArgV()[5] == -hugo", cmdLine.ArgV()[5], CLPS("-hugo"));
    AssertEqual("cmdLine.ArgV()[6] == 321 \"Heinz", cmdLine.ArgV()[6], CLPS("321 \"Heinz"));
    AssertEqual("cmdLine.ArgV()[7] == --tEst", cmdLine.ArgV()[7], CLPS("--tEst"));
    AssertEqual("cmdLine.ArgV()[8] == 4.2", cmdLine.ArgV()[8], CLPS("4.2"));
    AssertEqual("cmdLine.ArgV()[9] == Helmut", cmdLine.ArgV()[9], CLPS("Helmut"));
    AssertEqual("cmdLine.ArgV()[10] == -Help", cmdLine.ArgV()[10], CLPS("-Help"));
    AssertEqual("cmdLine.ArgV()[11] == 11", cmdLine.ArgV()[11], CLPS("11"));
    AssertEqual("cmdLine.ArgV()[12] == -ht", cmdLine.ArgV()[12], CLPS("-ht"));
    AssertEqual("cmdLine.ArgV()[13] == 4.1", cmdLine.ArgV()[13], CLPS("4.1"));

    CLParser parser;
    CLPOption helpOption(CLPS('h'), CLPS("help"), CLPS("Would print some helpful help."));
    CLPOption testOption(CLPS('t'), CLPS("Test"), NULL, CLPOption::FLAG_NULL, CLPOption::ValueDesc::ValueList(CLPOption::DOUBLE_OR_STRING_VALUE));

    AssertTrue("helpOption.GetValueCount() == 0", helpOption.GetValueCount() == 0);
    AssertTrue("testOption.GetValueCount() == 1", testOption.GetValueCount() == 1);
    AssertEqual("testOption.GetValueType(0) == DOUBLE_OR_STRING_VALUE", testOption.GetValueType(0), CLPOption::DOUBLE_OR_STRING_VALUE);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) == NULL", parser.FindOption(CLPS("HeLp")), NULL);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption(CLPS('t')), NULL);

    parser.AddOption(&helpOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption(CLPS("HeLp")), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption(CLPS('t')), NULL);

    parser.AddOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption(CLPS("HeLp")), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(H) == NULL", parser.FindOption(CLPS('H')), NULL);
    AssertEqual<const CLPOption*>("parser.FindOption(h) != NULL", parser.FindOption(CLPS('h')), &helpOption);
    AssertNotEqual<const CLPOption*>("parser.FindOption(t) != NULL", parser.FindOption(CLPS('t')), NULL);

    AssertException("Adding option a second time; Expecting IllegalParamException", parser.AddOption(&helpOption), vislib::IllegalParamException);

    parser.RemoveOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption(CLPS("HeLp")), &helpOption);
    AssertEqual<const CLPOption*>("parser.FindOption(t) == NULL", parser.FindOption(CLPS('t')), NULL);

    parser.AddOption(&testOption);

    AssertEqual<const CLPOption*>("parser.FindOption(HeLp) != NULL", parser.FindOption(CLPS("HeLp")), &helpOption);
    AssertNotEqual<const CLPOption*>("parser.FindOption(t) != NULL", parser.FindOption(CLPS('t')), NULL);

    bool parsed = AssertTrue("parser.Parse == 0", (parser.Parse(cmdLine.ArgC(), cmdLine.ArgV()) >= 0));
    parsed = true;

    CLParser::WarningIterator wii = parser.GetWarnings();
    if (wii.HasNext()) {
        printf("Checking Warnings:\n");
        while (wii.HasNext()) {
            CLPWarning &w = wii.Next();
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::YELLOW);
            printf("Warning %d [Arg %u]: ", int(w.GetWarnCode()), w.GetArgument());
            vislib::sys::Console::RestoreDefaultColors();
            printf("%s\n", CLParser::Warning::GetWarningString(w.GetWarnCode()));
        }
    }

    CLParser::ErrorIterator err = parser.GetErrors();
    if (err.HasNext()) {
        printf("Checking Errors:\n");
        while (err.HasNext()) {
            CLPError &e = err.Next();
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
            printf("Error %d [Arg %u]: ", int(e.GetErrorCode()), e.GetArgument());
            vislib::sys::Console::RestoreDefaultColors();
            printf("%s\n", e.GetErrorString(e.GetErrorCode()));
        }
        AssertOutput("Errors: ");
        AssertOutputFail(); // there should be no errors
    } else {
        AssertOutput("Errors: ");
        AssertOutputSuccess(); // there are no errors
    }

    if (parsed) {
        if (AssertEqual<unsigned int>("parser.ArgumentCount == 15", parser.ArgumentCount(), 15)) {
            bool suc = true;
            for (int i = 0; i < 15; i++) {
                char testtext[128];
                testtext[127] = 0;
#if (_MSC_VER >= 1400)
#pragma warning(disable: 4996)
#endif
#ifdef _WIN32
                _snprintf
#else
                snprintf
#endif
                    (testtext, 127, "Argument[%d] != NULL", i);
#if (_MSC_VER >= 1400)
#pragma warning(default: 4996)
#endif
                if (!AssertNotEqual<CLPArgument*>(testtext, parser.GetArgument(i), NULL)) {
                    suc = false;
                }
            }

            if (!AssertEqual<CLPArgument*>("Argument[15] == NULL", parser.GetArgument(15), NULL)) {
                suc = false;
            }

            if (suc) {
                AssertEqual("Argument[0]->Type == TYPE_UNKNOWN", parser.GetArgument(0)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[0]->option == NULL", parser.GetArgument(0)->GetOption(), NULL);
                AssertTrue("Argument[0]->IsSelected()", parser.GetArgument(0)->IsSelected());
                parser.GetArgument(0)->ToggleSelect();
                AssertFalse("! Argument[0]->IsSelected()", parser.GetArgument(0)->IsSelected());
                parser.GetArgument(0)->Select();
                AssertTrue("Argument[0]->IsSelected()", parser.GetArgument(0)->IsSelected());
                parser.GetArgument(0)->Deselect();
                AssertFalse("! Argument[0]->IsSelected()", parser.GetArgument(0)->IsSelected());
                parser.GetArgument(0)->ToggleSelect();
                AssertTrue("Argument[0]->IsSelected()", parser.GetArgument(0)->IsSelected());

                AssertEqual("Argument[1]->Type == TYPE_UNKNOWN", parser.GetArgument(1)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[1]->option == NULL", parser.GetArgument(1)->GetOption(), NULL);
                AssertTrue("Argument[1]->IsSelected()", parser.GetArgument(1)->IsSelected());

                AssertEqual("Argument[2]->Type == TYPE_UNKNOWN", parser.GetArgument(2)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[2]->option == NULL", parser.GetArgument(2)->GetOption(), NULL);
                AssertTrue("Argument[2]->IsSelected()", parser.GetArgument(2)->IsSelected());

                AssertEqual("Argument[3]->Type == TYPE_OPTION_SHORTNAMES", parser.GetArgument(3)->GetType(), CLPArgument::TYPE_OPTION_SHORTNAMES);
                AssertEqual<CLPOption*>("Argument[3]->option == &helpOption", parser.GetArgument(3)->GetOption(), &helpOption);
                AssertFalse("! Argument[3]->IsSelected()", parser.GetArgument(3)->IsSelected());

                AssertEqual("Argument[4]->Type == TYPE_UNKNOWN", parser.GetArgument(4)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[4]->option == NULL", parser.GetArgument(4)->GetOption(), NULL);
                AssertTrue("Argument[4]->IsSelected()", parser.GetArgument(4)->IsSelected());

                // normally arg[5] would be TYPE_OPTION_SHORTNAMES, but not all shortnames are known, so it must be unknown
                AssertEqual("Argument[5]->Type == TYPE_UNKNOWN", parser.GetArgument(5)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[5]->option == NULL", parser.GetArgument(5)->GetOption(), NULL);
                AssertTrue("Argument[5]->IsSelected()", parser.GetArgument(5)->IsSelected());

                AssertEqual("Argument[6]->Type == TYPE_UNKNOWN", parser.GetArgument(6)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[6]->option == NULL", parser.GetArgument(6)->GetOption(), NULL);
                AssertTrue("Argument[6]->IsSelected()", parser.GetArgument(6)->IsSelected());

                AssertEqual("Argument[7]->Type == TYPE_OPTION_LONGNAME", parser.GetArgument(7)->GetType(), CLPArgument::TYPE_OPTION_LONGNAME);
                AssertEqual<CLPOption*>("Argument[7]->option == &testOption", parser.GetArgument(7)->GetOption(), &testOption);
                AssertFalse("! Argument[7]->IsSelected()", parser.GetArgument(7)->IsSelected());

                AssertEqual("Argument[8]->Type == TYPE_OPTION_VALUE", parser.GetArgument(8)->GetType(), CLPArgument::TYPE_OPTION_VALUE);
                AssertEqual<CLPOption*>("Argument[8]->option == &testOption", parser.GetArgument(8)->GetOption(), &testOption);
                AssertFalse("! Argument[8]->IsSelected()", parser.GetArgument(8)->IsSelected());

                AssertEqual("Argument[9]->Type == TYPE_UNKNOWN", parser.GetArgument(9)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[9]->option == NULL", parser.GetArgument(9)->GetOption(), NULL);
                AssertTrue("Argument[9]->IsSelected()", parser.GetArgument(9)->IsSelected());

                // normally arg[10] would be TYPE_OPTION_SHORTNAMES, but NONE of the shortnames is known, so it must be unknown
                AssertEqual("Argument[10]->Type == TYPE_UNKNOWN", parser.GetArgument(10)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[10]->option == NULL", parser.GetArgument(10)->GetOption(), NULL);
                AssertTrue("Argument[10]->IsSelected()", parser.GetArgument(10)->IsSelected());

                AssertEqual("Argument[11]->Type == TYPE_UNKNOWN", parser.GetArgument(11)->GetType(), CLPArgument::TYPE_UNKNOWN);
                AssertEqual<CLPOption*>("Argument[11]->option == NULL", parser.GetArgument(11)->GetOption(), NULL);
                AssertTrue("Argument[11]->IsSelected()", parser.GetArgument(11)->IsSelected());

                AssertEqual("Argument[12]->Type == TYPE_OPTION_SHORTNAMES", parser.GetArgument(12)->GetType(), CLPArgument::TYPE_OPTION_SHORTNAMES);
                AssertEqual<CLPOption*>("Argument[12]->option == &helpOption", parser.GetArgument(12)->GetOption(), &helpOption);
                AssertFalse("! Argument[12]->IsSelected()", parser.GetArgument(12)->IsSelected());

                AssertEqual("Argument[13]->Type == TYPE_OPTION_SHORTNAMES", parser.GetArgument(13)->GetType(), CLPArgument::TYPE_OPTION_SHORTNAMES);
                AssertEqual<CLPOption*>("Argument[13]->option == &testOption", parser.GetArgument(13)->GetOption(), &testOption);
                AssertFalse("! Argument[13]->IsSelected()", parser.GetArgument(13)->IsSelected());

                AssertEqual("Argument[14]->Type == TYPE_OPTION_VALUE", parser.GetArgument(14)->GetType(), CLPArgument::TYPE_OPTION_VALUE);
                AssertEqual<CLPOption*>("Argument[14]->option == &testOption", parser.GetArgument(14)->GetOption(), &testOption);
                AssertFalse("! Argument[14]->IsSelected()", parser.GetArgument(14)->IsSelected());

                AssertEqual("First Help Option Occurrence is 3", helpOption.GetFirstOccurrence(), parser.GetArgument(3));
                AssertEqual("First Test Option Occurrence is 7", testOption.GetFirstOccurrence(), parser.GetArgument(7));

                CLPArgument *arg = parser.GetArgument(7);
                AssertException("Argument[7].GetValueString; Exception UnsupportedOperationException", arg->GetValueString(), vislib::UnsupportedOperationException);
                AssertException("Argument[7].GetValueInt; Exception UnsupportedOperationException", arg->GetValueInt(), vislib::UnsupportedOperationException);
                AssertNoException("Argument[7].GetValueDouble; Exception UnsupportedOperationException", arg->GetValueDouble());
                AssertNearlyEqual("Argument[7].GetValueDouble == 4.2", arg->GetValueDouble(), 4.2);

                arg = parser.GetArgument(8);
                AssertException("Argument[8].GetValueString; Exception UnsupportedOperationException", arg->GetValueString(), vislib::UnsupportedOperationException);
                AssertException("Argument[8].GetValueInt; Exception UnsupportedOperationException", arg->GetValueInt(), vislib::UnsupportedOperationException);
                AssertNoException("Argument[8].GetValueDouble; Exception UnsupportedOperationException", arg->GetValueDouble());
                AssertNearlyEqual("Argument[8].GetValueDouble == 4.2", arg->GetValueDouble(), 4.2);

                arg = parser.GetArgument(13);
                AssertException("Argument[13].GetValueString; Exception UnsupportedOperationException", arg->GetValueString(), vislib::UnsupportedOperationException);
                AssertException("Argument[13].GetValueInt; Exception UnsupportedOperationException", arg->GetValueInt(), vislib::UnsupportedOperationException);
                AssertNoException("Argument[13].GetValueDouble; Exception UnsupportedOperationException", arg->GetValueDouble());
                AssertNearlyEqual("Argument[13].GetValueDouble == 4.1", arg->GetValueDouble(), 4.1);

                arg = parser.GetArgument(14);
                AssertException("Argument[14].GetValueString; Exception UnsupportedOperationException", arg->GetValueString(), vislib::UnsupportedOperationException);
                AssertException("Argument[14].GetValueInt; Exception UnsupportedOperationException", arg->GetValueInt(), vislib::UnsupportedOperationException);
                AssertNoException("Argument[14].GetValueDouble; Exception UnsupportedOperationException", arg->GetValueDouble());
                AssertNearlyEqual("Argument[14].GetValueDouble == 4.1", arg->GetValueDouble(), 4.1);

                CLProvider selectedArgs;
                parser.ExtractSelectedArguments(selectedArgs);

                if (AssertEqual("selectedArgs.ArgC == 9", selectedArgs.ArgC(), 9)) {
                    AssertEqual("selectedArgs.ArgV[0] == ModuleName", selectedArgs.ArgV()[0], cmdName);
                    AssertEqual("selectedArgs.ArgV[1] == Horst", selectedArgs.ArgV()[1], CLPS("Horst"));
                    AssertEqual("selectedArgs.ArgV[2] == 1.2", selectedArgs.ArgV()[2], CLPS("1.2"));
                    AssertEqual("selectedArgs.ArgV[3] == <empty String>", selectedArgs.ArgV()[3], CLPS(""));
                    AssertEqual("selectedArgs.ArgV[4] == -hugo", selectedArgs.ArgV()[4], CLPS("-hugo"));
                    AssertEqual("selectedArgs.ArgV[5] == 321 \"Heinz", selectedArgs.ArgV()[5], CLPS("321 \"Heinz"));
                    AssertEqual("selectedArgs.ArgV[6] == Helmut", selectedArgs.ArgV()[6], CLPS("Helmut"));
                    AssertEqual("selectedArgs.ArgV[7] == -Help", selectedArgs.ArgV()[7], CLPS("-Help"));
                    AssertEqual("selectedArgs.ArgV[8] == 11", selectedArgs.ArgV()[8], CLPS("11"));

                    AssertEqual("selectedArgs.SingleCmdLine correct", 
                        selectedArgs.SingleStringCommandLine(false),
                        CLPS("Horst 1.2 \"\" -hugo \"321 \"\"Heinz\" Helmut -Help 11"));
                        
                }

            }
        }
    }

    printf("Testing generated online help:\n");
    CLParser::OptionDescIterator odi = parser.OptionDescriptions();
    if (AssertEqual("first description block present", odi.HasNext(), true)) {
        CLPChar *desc = odi.Next();
        printf("%s\n", desc);
        AssertEqual("first description block correct", desc, CLPS("--help -h  Would print some helpful help."));
        if (AssertEqual("second description block present", odi.HasNext(), true)) {
            desc = odi.Next();
            printf("%s\n", desc);
            AssertEqual("second description block correct", desc, CLPS("--Test -t"));
            AssertEqual("no third description block available", odi.HasNext(), false);
        }
    }

    { // new test on options with multiple values 
        CLProvider cmdLine(cmdName, CLPS("Horst -t 12 1.2 true ninja -t Hugo 1 2 3 4 5 6 7 8 9 Ende"));
        CLParser parser;
        CLPOption testOption(CLPS('t'), CLPS("Test"), CLPS("This is a rather stupid test option with multiple, four to be exact, values of defined types."
            "The option is not used to do something interesting, but the parser is tested."), CLPOption::FLAG_UNIQUE, 
            CLPOption::ValueDesc::ValueList(CLPOption::INT_VALUE, "Count", "Number of Stuff or something")
            ->Add(CLPOption::DOUBLE_OR_STRING_VALUE, "Factor", "Increases or decreases the size of Stuff")
            ->Add(CLPOption::BOOL_OR_STRING_VALUE, "Flag", "May activate something.")
            ->Add(CLPOption::STRING_VALUE, "Desc", "Finallay some real words"));
        parser.AddOption(&testOption);

        parsed = AssertTrue("parser.Parse == 0", (parser.Parse(cmdLine.ArgC(), cmdLine.ArgV()) >= 0));

        CLParser::WarningIterator wii = parser.GetWarnings();
        if (wii.HasNext()) {
            printf("Checking Warnings:\n");
            while (wii.HasNext()) {
                CLPWarning &w = wii.Next();
                vislib::sys::Console::SetForegroundColor(vislib::sys::Console::YELLOW);
                printf("Warning %d [Arg %u]: ", int(w.GetWarnCode()), w.GetArgument());
                vislib::sys::Console::RestoreDefaultColors();
                printf("%s\n", CLParser::Warning::GetWarningString(w.GetWarnCode()));
            }
        }

        CLParser::ErrorIterator err = parser.GetErrors();
        if (err.HasNext()) {
            printf("Checking Errors (EXPECTING 1):\n");
            while (err.HasNext()) {
                CLPError &e = err.Next();
                vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
                printf("Error %d [Arg %u]: ", int(e.GetErrorCode()), e.GetArgument());
                vislib::sys::Console::RestoreDefaultColors();
                printf("%s\n", e.GetErrorString(e.GetErrorCode()));
            }
        }

        AssertTrue("testOption found", testOption.GetFirstOccurrence() != NULL);

        AssertEqual<unsigned int>("Parsing resulted into 19 Argument objects", parser.ArgumentCount(), 19);
        //if (parser.ArgumentCount() == 19) {
        //    for (int i = 0; i < 19; i++) {
        //        CLParser::Argument *arg = parser.GetArgument(i);
        //        printf("%i: \"%s\" %i %i\n", i, arg->GetInputString(), int(arg->GetType()), int(arg->GetValueType()));
        //    }
        //}

        CLParser::OptionDescIterator odi = parser.OptionDescriptions(true);
        while (odi.HasNext()) {
            printf("%s", odi.Next());
        }

    }
}


/*
 * Test of the command line parser.
 */
void TestCmdLineParser2(void) {
    /** the parser object */
    vislib::sys::TCmdLineParser parser;
    /** the parser help option */
    vislib::sys::TCmdLineParser::Option help(_T('h'), _T("help"), _T("Prints this help message."),
        vislib::sys::TCmdLineParser::Option::FLAG_EXCLUSIVE);
    /** parser option of the amber input files */
    vislib::sys::TCmdLineParser::Option inputFiles(_T('i'), _T("input"), 
        _T("Required! Specifies the two input files (Amber topology, netcdf trajectory)."), 
        vislib::sys::TCmdLineParser::Option::FLAG_UNIQUE | vislib::sys::TCmdLineParser::Option::FLAG_REQUIRED, 
        vislib::sys::TCmdLineParser::Option::ValueDesc::ValueList(
            vislib::sys::TCmdLineParser::Option::STRING_VALUE, _T("TopFile"), _T("The Amber topology file"))->Add(
            vislib::sys::TCmdLineParser::Option::STRING_VALUE, _T("NetCDF"), _T("The netcdf trajectory file")));
    /** parser option of the pathline output file */
    vislib::sys::TCmdLineParser::Option outputFile(_T('o'), _T("output"), 
        _T("Required! Specifies the output file (solvent pathlines)."),
        vislib::sys::TCmdLineParser::Option::FLAG_UNIQUE | vislib::sys::TCmdLineParser::Option::FLAG_REQUIRED, 
        vislib::sys::TCmdLineParser::Option::ValueDesc::ValueList(
            vislib::sys::TCmdLineParser::Option::STRING_VALUE, _T("SolPathFile"), _T("The solvent pathlines file")));

    parser.AddOption(&help);
    parser.AddOption(&inputFiles);
    parser.AddOption(&outputFile);

    vislib::sys::TCmdLineProvider cmdLine(
  _T("DUMMYAPP.exe -i //SFB716/Datensaetze/TEM1_wt.top //SFB716/Datensaetze/TEM1_wt-1001-rmsfit.bintrj --output //bossanova/bidmon/SFB716/Datensaetze/TEM1_wt-1001.solpath")
  //_T("DUMMYAPP.exe -i 1 //i -o 2") // FAILS!
  //_T("DUMMYAPP.exe -i 1 //o -o 2") // FAILS!
  //_T("DUMMYAPP.exe -i 1 //h -o 2") // Works!
  //_T("DUMMYAPP.exe -i \\\\bossanova\\bidmon\\SFB716\\Datensaetze\\TEM1_wt.top \\\\bossanova\\bidmon\\SFB716\\Datensaetze\\TEM1_wt-1001-rmsfit.bintrj --output \\\\bossanova\\bidmon\\SFB716\\Datensaetze\\TEM1_wt-1001.solpath")
  //_T("DUMMYAPP.exe -i 1 \\\\i -o 2") // FAILS!
  //_T("DUMMYAPP.exe -i 1 \\\\o -o 2") // FAILS!
  //_T("DUMMYAPP.exe -i 1 \\\\h -o 2") // Works!
  //_T("DUMMYAPP.exe -i 1 /io -o 2") // FAILS!
  //_T("DUMMYAPP.exe -i 1 abi -o 2") // FAILS!
        );

    parser.Parse(cmdLine.ArgC(), cmdLine.ArgV(), false);

    AssertTrue("No Parser Errors", parser.GetErrors().HasNext());
    vislib::sys::TCmdLineParser::ErrorIterator errorIter = parser.GetErrors();
    while (errorIter.HasNext()) {
        printf("%s\n", vislib::sys::TCmdLineParser::Error::GetErrorString(
            errorIter.Next().GetErrorCode()));
    }
    printf("\n");

    AssertTrue("No Parser Warnings", parser.GetWarnings().HasNext());
    vislib::sys::TCmdLineParser::WarningIterator warnIter = parser.GetWarnings();
    while (warnIter.HasNext()) {
        printf("%s\n", vislib::sys::TCmdLineParser::Warning::GetWarningString(
            warnIter.Next().GetWarnCode()));
    }
    printf("\n");

    printf("Arguments:\n");
    for (unsigned int idx = 0; idx < parser.ArgumentCount(); idx++) {
        vislib::sys::TCmdLineParser::Argument *arg = parser.GetArgument(idx);
        if (arg == NULL) {
            printf("NULL\n");
            continue;
        }
        switch (arg->GetType()) {
            case vislib::sys::TCmdLineParser::Argument::TYPE_UNKNOWN : 
                printf("TYPE_UNKNOWN ");
                break;
            case vislib::sys::TCmdLineParser::Argument::TYPE_OPTION_LONGNAME : 
                printf("TYPE_OPTION_LONGNAME ");
                break;
            case vislib::sys::TCmdLineParser::Argument::TYPE_OPTION_SHORTNAMES : 
                printf("TYPE_OPTION_SHORTNAMES ");
                break;
            case vislib::sys::TCmdLineParser::Argument::TYPE_OPTION_VALUE : 
                printf("TYPE_OPTION_VALUE ");
                break;
        }
        printf("%s ", T2A(arg->GetInputString()));
        if (arg->IsSelected()) {
            printf("[Selected]");
        }
        printf("\n");
    }

}


/*
 * Test of the command line parser.
 */
void TestCmdLineParser(void) {
    TestCmdLineParser1();
    //TestCmdLineParser2();
}
