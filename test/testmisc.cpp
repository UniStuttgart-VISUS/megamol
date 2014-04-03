/*
 * testmisc.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testmisc.h"
#include "testhelper.h"

#include "vislib/Array.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/Console.h"
#include "vislib/ConsoleProgressBar.h"
#include "vislib/ColumnFormatter.h"
#include "the/exception.h"
#include "the/system/system_exception.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/SingleLinkedList.h"
#include "the/text/string_converter.h"
#include "the/system/system_message.h"
#include "vislib/sysfunctions.h"
#include "vislib/Path.h"
#include "the/trace.h"
#include "vislib/FileNameSequence.h"
#include "vislib/RawStorage.h"
#include "vislib/NamedColours.h"
#include "vislib/ColourParser.h"
#include "the/utils.h"

// #define USE_UNICODE_COLUMNFORMATTER

#ifndef USE_UNICODE_COLUMNFORMATTER
typedef vislib::ColumnFormatterA ColumnFormatter;
typedef the::astring CFString;
typedef char CFChar;
#define CFS(A) the::astring(A)

#else // USE_UNICODE_COLUMNFORMATTER
typedef vislib::ColumnFormatterW ColumnFormatter;
typedef the::wstring CFString;
typedef wchar_t CFChar;
#define CFS(A) the::wstring(L ## A)

#endif // USE_UNICODE_COLUMNFORMATTER

void TestConsoleColours(void) {
    bool hc = vislib::sys::Console::ColorsAvailable();
    printf("Console has %scolors\n", hc ? "" : "no ");
    if (!hc) return;

    printf("Colors are %sabled\n", vislib::sys::Console::ColorsEnabled() ? "en" : "dis");

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::DARK_RED);
    printf("Text set to Dark Red on Default\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    printf("Text set to Red on Default\n");
    fprintf(stdout, "Text to stdout\n");
    fprintf(stderr, "Text to stderr\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::DARK_RED);
    printf("Text set to Dark Red on Default\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::SetBackgroundColor(vislib::sys::Console::GREEN);
    printf("Text set to Red on Green\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::SetBackgroundColor(vislib::sys::Console::DARK_BLUE);
    printf("Text set to Red on Dark Blue\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::DARK_MAGENTA);
    vislib::sys::Console::SetBackgroundColor(vislib::sys::Console::BLACK);
    printf("Text set to Dark Magenta on Black\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::RestoreDefaultColors();
    printf("Text set to Default on Default\n");
    fprintf(stdout, "Text to stdout\n");
    fprintf(stderr, "Text to stderr\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::EnableColors(false);
    printf("Disableing colors\n");
    printf("Colors are %sabled\n", vislib::sys::Console::ColorsEnabled() ? "en" : "dis");

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::BLUE);
    printf("Text set to Blue on Default (should stay Default on Default)\n");
    fprintf(stdout, "Text to stdout\n");
    fprintf(stderr, "Text to stderr\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::RestoreDefaultColors();
    printf("Text set to Default on Default\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::EnableColors(true);
    printf("Enableing colors\n");
    printf("Colors are %sabled\n", vislib::sys::Console::ColorsEnabled() ? "en" : "dis");

    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::BLUE);
    printf("Text set to Blue on Default (should stay Default on Default)\n");
    fprintf(stdout, "Text to stdout\n");
    fprintf(stderr, "Text to stderr\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    vislib::sys::Console::RestoreDefaultColors();
    printf("Text set to Default on Default\n");
    fprintf(stdout, "Text to stdout\n");
    fprintf(stderr, "Text to stderr\n");
    printf("Foregound Color is %d\n", vislib::sys::Console::GetForegroundColor());
    printf("Backgound Color is %d\n", vislib::sys::Console::GetBackgroundColor());

    for (int i = 0; i < 16; i++) {
        printf("Color: ");
        vislib::sys::Console::SetForegroundColor(static_cast<vislib::sys::Console::ColorType>(i));
        for (int j = 0; j < 16; j++) {
            vislib::sys::Console::SetBackgroundColor(static_cast<vislib::sys::Console::ColorType>(j));
            printf(" %x%x", i, j);
        }
        vislib::sys::Console::RestoreDefaultColors();
        printf("\n");
    }
    
    printf("The Console's Size: %u x %u\n", vislib::sys::Console::GetWidth(), vislib::sys::Console::GetHeight());

}

void TestColumnFormatterHelper(CFString &result, CFString &expected) {
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    std::cout << "Expected:" << std::endl;
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << expected.c_str() << std::endl;
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    std::cout << "Result:" << std::endl;
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << result.c_str() << std::endl;
}

void TestColumnFormatter(void) {
    ColumnFormatter ColFormatter;
    CFString output = CFS("Test");
    CFString expected;

    ColFormatter.SetMaxWidth(79);
    AssertEqual<unsigned int>("Maximum Width = 79", ColFormatter.GetMaxWidth(), 79);

    ColFormatter.SetColumnCount(4);
    AssertEqual<unsigned int>("Column Count = 4", ColFormatter.GetColumnCount(), 4);

    ColFormatter.SetSeparator(CFS(" | "));
    AssertEqual("Separator = \" | \"", ColFormatter.GetSeparator(), CFS(" | "));

    ColFormatter[0].SetWidth(6);
    AssertEqual<unsigned int>("Column[0] Width = 6", ColFormatter[0].GetWidth(), 6);
    ColFormatter[0].DisableWrapping();
    AssertEqual("Column[0] Wrapping = false", ColFormatter[0].IsWrappingDisabled(), true);

    ColFormatter[1].SetWidth(20);
    AssertEqual<unsigned int>("Column[1] Width = 20", ColFormatter[1].GetWidth(), 20);
    ColFormatter[1].EnableWrapping();
    AssertEqual("Column[1] Wrapping = true", ColFormatter[1].IsWrappingDisabled(), false);

    ColFormatter[2].SetWidth(24);
    AssertEqual<unsigned int>("Column[2] Width = 24", ColFormatter[2].GetWidth(), 24);
    AssertEqual("Column[2] Wrapping = true", ColFormatter[2].IsWrappingDisabled(), false);

    ColFormatter[3].SetWidth(0);
    AssertEqual<unsigned int>("Column[3] Width = 0", ColFormatter[3].GetWidth(), 0);
    ColFormatter[3].DisableWrapping();
    AssertEqual("Column[3] Wrapping = false", ColFormatter[3].IsWrappingDisabled(), true);

    ColFormatter[0].SetText(CFS("1.  "));
    ColFormatter[1].SetText(CFS("This is 1. test"));
    ColFormatter[2].SetText(CFS("All Text fits into line"));
    ColFormatter[3].SetText(CFS("No wrapping at all"));
    
    AssertEqual("Column[0] Text = \"1.\"", ColFormatter[0].GetText(), CFS("1."));
    AssertEqual("Column[1] Text = \"This is 1. test\"", ColFormatter[1].GetText(), CFS("This is 1. test"));
    AssertEqual("Column[2] Text = \"All Text fits into line\"", ColFormatter[2].GetText(), CFS("All Text fits into line"));
    AssertEqual("Column[3] Text = \"No wrapping at all\"", ColFormatter[3].GetText(), CFS("No wrapping at all"));

    ColFormatter.SetColumnCount(3);
    AssertEqual<unsigned int>("Column Count = 3", ColFormatter.GetColumnCount(), 3);

    AssertEqual("Separator = \" | \"", ColFormatter.GetSeparator(), CFS(" | "));

    AssertEqual<unsigned int>("Column[0] Width = 6", ColFormatter[0].GetWidth(), 6);
    AssertEqual("Column[0] Wrapping = false", ColFormatter[0].IsWrappingDisabled(), true);

    AssertEqual<unsigned int>("Column[1] Width = 20", ColFormatter[1].GetWidth(), 20);
    AssertEqual("Column[1] Wrapping = true", ColFormatter[1].IsWrappingDisabled(), false);

    AssertEqual<unsigned int>("Column[2] Width = 24", ColFormatter[2].GetWidth(), 24);
    AssertEqual("Column[2] Wrapping = true", ColFormatter[2].IsWrappingDisabled(), false);

    AssertEqual("Column[0] Text = \"1.\"", ColFormatter[0].GetText(), CFS("1."));
    AssertEqual("Column[1] Text = \"This is 1. test\"", ColFormatter[1].GetText(), CFS("This is 1. test"));
    AssertEqual("Column[2] Text = \"All Text fits into line\"", ColFormatter[2].GetText(), CFS("All Text fits into line"));

    AssertException("Accessing Column[3]: InvalidParamException", ColFormatter[3].SetWidth(0), the::argument_exception);

    ColFormatter.SetColumnCount(4);
    AssertEqual<unsigned int>("Column Count = 4", ColFormatter.GetColumnCount(), 4);

    AssertEqual("Separator = \" | \"", ColFormatter.GetSeparator(), CFS(" | "));

    AssertEqual<unsigned int>("Column[0] Width = 6", ColFormatter[0].GetWidth(), 6);
    AssertEqual("Column[0] Wrapping = false", ColFormatter[0].IsWrappingDisabled(), true);

    AssertEqual<unsigned int>("Column[1] Width = 20", ColFormatter[1].GetWidth(), 20);
    AssertEqual("Column[1] Wrapping = true", ColFormatter[1].IsWrappingDisabled(), false);

    AssertEqual<unsigned int>("Column[2] Width = 24", ColFormatter[2].GetWidth(), 24);
    AssertEqual("Column[2] Wrapping = true", ColFormatter[2].IsWrappingDisabled(), false);

    ColFormatter[3].SetWidth(0);
    AssertEqual<unsigned int>("Column[3] Width = 0", ColFormatter[3].GetWidth(), 0);
    ColFormatter[3].DisableWrapping();
    AssertEqual("Column[3] Wrapping = false", ColFormatter[3].IsWrappingDisabled(), true);

    ColFormatter[3].SetText(CFS("No wrapping at all"));
    
    AssertEqual("Column[0] Text = \"1.\"", ColFormatter[0].GetText(), CFS("1."));
    AssertEqual("Column[1] Text = \"This is 1. test\"", ColFormatter[1].GetText(), CFS("This is 1. test"));
    AssertEqual("Column[2] Text = \"All Text fits into line\"", ColFormatter[2].GetText(), CFS("All Text fits into line"));
    AssertEqual("Column[3] Text = \"No wrapping at all\"", ColFormatter[3].GetText(), CFS("No wrapping at all"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
    expected = CFS("1.     | This is 1. test      | All Text fits into line  | No wrapping at all");
    if (!AssertEqual("Formatted 1. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
//                               123456
    ColFormatter[0].SetText(CFS("Second"));
//                               12345678901234567890
    ColFormatter[1].SetText(CFS("The second test comes with some word wraps."));
//                               123456789012345678901234
    ColFormatter[2].SetText(CFS("But no out of columns"));
//                               12345678901234567890
    ColFormatter[3].SetText(CFS("So the wrapping is very easy."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
    expected = CFS("Second | The second test      | But no out of columns    | So the wrapping is\n")
             + CFS("       | comes with some word |                          | very easy.\n")
             + CFS("       | wraps.               |                          |");
    if (!AssertEqual("Formatted 2. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
//                               123456
    ColFormatter[0].SetText(CFS("The Third Test"));
//                               12345678901234567890
    ColFormatter[1].SetText(CFS("TTT performs some out of column tests."));
//                               123456789012345678901234
    ColFormatter[2].SetText(CFS("A short out of column test!"));
//                               12345678901234567890
    ColFormatter[3].SetText(CFS("Seeing forward to test 4."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
    expected = CFS("The Third Test | TTT performs | A short out of column    | Seeing forward to\n")
             + CFS("       | some out of column   | test!                    | test 4.\n")
             + CFS("       | tests.               |                          |");
    if (!AssertEqual("Formatted 3. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
//                               123456
    ColFormatter[0].SetText(CFS("The fourth test comes with a very very very very very very very very very very very long first column as hardcore test."));
//                               12345678901234567890
    ColFormatter[1].SetText(CFS("2. Column is in 3. line"));
//                               123456789012345678901234
    ColFormatter[2].SetText(CFS("The 3. Column starts where it should be."));
//                               12345678901234567890
    ColFormatter[3].SetText(CFS("This was the fourth test."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
    expected = CFS("The fourth test comes with a very very very very very very very very very very\n")
             + CFS("very long first column as hardcore test. | The 3. Column | This was the fourth\n")
             + CFS("       | 2. Column is in 3.   | starts where it should   | test.\n")
             + CFS("       | line                 | be.                      |");
    if (!AssertEqual("Formatted 4. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter.SetColumnCount(3);
    ColFormatter.SetSeparator(CFS(".--."));
    ColFormatter[0].SetWidth(0);
    ColFormatter[0].EnableWrapping();
    ColFormatter[0].SetText(CFS("01234567890123456789")); // 20 characters
    ColFormatter[1].SetWidth(30);
    ColFormatter[1].EnableWrapping();
    ColFormatter[1].SetText(CFS("012345678901234567890123456789"));
    ColFormatter[2].SetWidth(0);
    ColFormatter[2].EnableWrapping();
    ColFormatter[2].SetText(CFS("012345678901234567890123456789"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
             + CFS("                    .--.                              .--.123456789");
    if (!AssertEqual("Formatted 5. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[0].SetWidth(102);

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
             + CFS("                    .--.                              .--.123456789");
    if (!AssertEqual("Formatted 6. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].DisableWrapping();
    ColFormatter[1].SetText(CFS("012345678901234567890123456789012345678901234567890123456789"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.0123456789012345678901234567890123456789012345678901234\n")
             + CFS("                    .--.56789                         .--.012345678901234567890\n")
             + CFS("                    .--.                              .--.123456789");
    if (!AssertEqual("Formatted 7. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].EnableWrapping();

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
             + CFS("                    .--.012345678901234567890123456789.--.123456789");
    if (!AssertEqual("Formatted 8. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[0].SetWidth(20);
    ColFormatter[0].DisableWrapping();
    ColFormatter[0].SetText(CFS("012345678901234567890123456789012345678901234567890"));
    ColFormatter[1].SetText(CFS("Horst"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("012345678901234567890123456789012345678901234567890   .--.012345678901234567890\n")
             + CFS("                    .--.Horst                         .--.123456789");
    if (!AssertEqual("Formatted 9. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetWidth(0);
    ColFormatter[0].SetText(CFS("First Column"));
    ColFormatter[2].SetText(CFS("Third Column"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.Horst.--.Third Column");
    if (!AssertEqual("Formatted 10. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column now gets a very very long text to push into a new line."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.The second column now gets a very very long text to\n")
             + CFS("                    .--.push into a new line..--.Third Column");
    if (!AssertEqual("Formatted 11. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column is now shorter, But still too long"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.The second column is now shorter, But still too long\n")
             + CFS("                    .--..--.Third Column");
    if (!AssertEqual("Formatted 12. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column is now a bit longer again.  I want to check space interpretation on word wrap here."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.The second column is now a bit longer again.  I want to\n")
             + CFS("                    .--.check space interpretation on word wrap here..--.Third\n")
             + CFS("                    .--.                                             .--.Column");
    if (!AssertEqual("Formatted 13. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

}


void TestExceptions(void) {
    the::system::system_exception e1(2, __FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e1.what());

    the::exception e2(__FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e2.what()); 
}

void Test_system_message(void) {
    the::system::system_message sysMsg(4);
    ::_tprintf(_T("%s\n"), static_cast<const TCHAR *>(sysMsg));
}

void TestPerformanceCounter(void) {
    for (int i = 0; i < 100; i++) {
        ::_tprintf(_T("%lu\n"), (unsigned long)vislib::sys::PerformanceCounter::Query(true));
    ::_tprintf(_T("%f\n"), vislib::sys::PerformanceCounter::QueryMillis());
        int c = 0;
        for (int j = 0; j < 1000; j++) {
            for (int k = 0; k < j; k++) {
                c += k;
            }
        }
        c = c / 2;
    }
}

void TestPathManipulations(void) {
    // Hazard! Think of some more secure test cases.

    //try {
    //    vislib::sys::Path::MakeDirectory(L"Horst/Hugo/Heinz/Hans/Helmut");
    //    vislib::sys::Path::DeleteDirectory("Wurst", true);
    //} catch(the::system::system_exception e) {
    //    fprintf(stderr, "the::system::system_exception: %s\n", e.what());
    //} catch(the::exception e) {
    //    fprintf(stderr, "Exception: %s\n", e.what());
    //} catch(...) {
    //    fprintf(stderr, "Unknown Exception.\n");
    //}
}


/*
 * TestFileNameSequence
 */
void TestFileNameSequence(void) {
    using vislib::sys::FileNameSequence;
    using vislib::SmartPtr;

    FileNameSequence seq;
    FileNameSequence::FileNameStringElementA *s1
        = new FileNameSequence::FileNameStringElementA();
    FileNameSequence::FileNameCountElement *c1
        = new FileNameSequence::FileNameCountElement();
    FileNameSequence::FileNameStringElementA *s2
        = new FileNameSequence::FileNameStringElementA();
    FileNameSequence::FileNameCountElement *c2
        = new FileNameSequence::FileNameCountElement();
    FileNameSequence::FileNameStringElementA *s3
        = new FileNameSequence::FileNameStringElementA();

    s1->SetText("C:\\Some\\Directory\\AndFileName");
    c1->SetRange(1, 2);
    c1->SetDigits(2);
    c1->ResetCounter();
    AssertEqual<unsigned int>("counter1::count == 2", c1->Count(), 2);
    s2->SetText("_");
    c2->SetRange(0, 10, 3);
    c2->SetDigits(4);
    c2->ResetCounter();
    AssertEqual<unsigned int>("counter2::count == 4", c2->Count(), 4);
    s3->SetText(".dmy");

    seq.SetElementCount(5);
    AssertFalse("Sequence is not valid yet", seq.IsValid());
    seq.SetElement(0, s1);
    seq.SetElement(1, c1);
    seq.SetElement(2, s2);
    seq.SetElement(3, c2);
    seq.SetElement(4, s3);
    AssertTrue("Sequence is valid", seq.IsValid());
    AssertEqual<unsigned int>("Sequence::count == 8", seq.Count(), 8);

    AssertFalse("Sequence is in normal order", seq.ReversedCounterPriority());

    AssertEqual("File 0 is correct", seq.FileNameA(0).c_str(), "C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 0 is correct", seq.FileNameW(0).c_str(), L"C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 1 is correct", seq.FileNameA(1).c_str(), "C:\\Some\\Directory\\AndFileName01_0003.dmy");
    AssertEqual("File 2 is correct", seq.FileNameA(2).c_str(), "C:\\Some\\Directory\\AndFileName01_0006.dmy");
    AssertEqual("File 3 is correct", seq.FileNameA(3).c_str(), "C:\\Some\\Directory\\AndFileName01_0009.dmy");
    AssertEqual("File 4 is correct", seq.FileNameA(4).c_str(), "C:\\Some\\Directory\\AndFileName02_0000.dmy");
    AssertEqual("File 5 is correct", seq.FileNameA(5).c_str(), "C:\\Some\\Directory\\AndFileName02_0003.dmy");
    AssertEqual("File 6 is correct", seq.FileNameA(6).c_str(), "C:\\Some\\Directory\\AndFileName02_0006.dmy");
    AssertEqual("File 7 is correct", seq.FileNameA(7).c_str(), "C:\\Some\\Directory\\AndFileName02_0009.dmy");

    seq.SetReversedCounterPriority(true);
    AssertTrue("Sequence is in reversed order", seq.ReversedCounterPriority());

    AssertEqual("File 0 is correct", seq.FileNameA(0).c_str(), "C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 1 is correct", seq.FileNameA(1).c_str(), "C:\\Some\\Directory\\AndFileName02_0000.dmy");
    AssertEqual("File 2 is correct", seq.FileNameA(2).c_str(), "C:\\Some\\Directory\\AndFileName01_0003.dmy");
    AssertEqual("File 3 is correct", seq.FileNameA(3).c_str(), "C:\\Some\\Directory\\AndFileName02_0003.dmy");
    AssertEqual("File 4 is correct", seq.FileNameA(4).c_str(), "C:\\Some\\Directory\\AndFileName01_0006.dmy");
    AssertEqual("File 5 is correct", seq.FileNameA(5).c_str(), "C:\\Some\\Directory\\AndFileName02_0006.dmy");
    AssertEqual("File 6 is correct", seq.FileNameA(6).c_str(), "C:\\Some\\Directory\\AndFileName01_0009.dmy");
    AssertEqual("File 7 is correct", seq.FileNameA(7).c_str(), "C:\\Some\\Directory\\AndFileName02_0009.dmy");

    // no need to clean up s1, s2, s3, c1, and c2 because they have been
    // assigned to smart pointers ;-)

    /*
    seq.Autodetect("T:\\grottel\\Photos\\HotelRoom1\\DSC00004.JPG");
    if (seq.IsValid()) {
        unsigned int cnt = seq.Count();
        printf("Autodetected %u files:\n", cnt);
        for (unsigned int i = 0; i < cnt; i++) {
            printf("  %s\n", seq.FileNameA(i).c_str());
        }
    } else {
        printf("Autodetection failed (Sequence is invalid).\n");
    }
    //*/
}


/*
 * TestAsciiFile
 */
void TestAsciiFile(void) {
#define KLASSIK_TEST 1
#ifdef KLASSIK_TEST
    const char *filename = "D:\\data\\random.imd";
#else
    const char *filename = "C:\\VIS\\data\\demo.siff";
#endif
    vislib::sys::ASCIIFileBuffer afb;
    vislib::sys::PerformanceCounter pc(true);

    pc.SetMark();
    AssertTrue("Loading text file", afb.LoadFile(filename));
    uint64_t e = pc.Difference();
#ifdef KLASSIK_TEST
    AssertEqual<size_t>("All lines loaded", afb.Count(), 40000008UL);
#else
    AssertEqual<size_t>("All lines loaded", afb.Count(), 17UL);
#endif
    //printf("%d\n", afb.Count());
    printf("Loaded in %f seconds\n", pc.ToMillis(e) / 1000.0);

#ifdef KLASSIK_TEST
    AssertEqual<size_t>("Length of line 0 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[0])), 17);
    AssertEqual<size_t>("Length of line 1 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[1])), 9);
    AssertEqual<size_t>("Length of line 2 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[2])), 11);
    AssertEqual<size_t>("Length of line 3 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[3])), 11);
    AssertEqual<size_t>("Length of line 4 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[4])), 11);
    AssertEqual<size_t>("Length of line 5 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[5])), 76);
    AssertEqual<size_t>("Length of line 6 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[6])), 3);
    AssertEqual<size_t>("Length of line 7 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[7])), 49);
    AssertEqual<size_t>("Length of line 8 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[8])), 48);
    AssertEqual<size_t>("Length of line 9 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[9])), 48);
#else
    AssertEqual<size_t>("Length of line 0 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[0])), 9);
    AssertEqual<size_t>("Length of line 1 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[1])), 21);
    AssertEqual<size_t>("Length of line 2 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[2])), 17);
    AssertEqual<size_t>("Length of line 3 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[3])), 17);
    AssertEqual<size_t>("Length of line 4 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[4])), 17);
    AssertEqual<size_t>("Length of line 5 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[5])), 22);
    AssertEqual<size_t>("Length of line 6 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[6])), 22);
    AssertEqual<size_t>("Length of line 7 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[7])), 22);
    AssertEqual<size_t>("Length of line 8 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[8])), 19);
    AssertEqual<size_t>("Length of line 9 correct", the::text::string_utility::c_str_len(static_cast<const char*>(afb[9])), 20);
#endif

#ifdef KLASSIK_TEST
    vislib::sys::MemmappedFile file;
    AssertTrue("Opening file again", file.Open(filename,
        vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY));
    pc.SetMark();
    while (!file.IsEOF()) {
        vislib::sys::ReadLineFromFileA(file);
    }
    e = pc.Difference();
    printf("Manually loaded in %f seconds\n", pc.ToMillis(e) / 1000.0);
#else
    pc.SetMark();
    AssertTrue("Loading text file again (word-wise)", afb.LoadFile(filename, vislib::sys::ASCIIFileBuffer::PARSING_WORDS));
    e = pc.Difference();
    printf("Loaded (word-wise) in %f seconds\n", pc.ToMillis(e) / 1000.0);
    AssertEqual<size_t>("Still all lines loaded", afb.Count(), 17UL);
    AssertEqual<size_t>("Line 0 has 2 words", afb[0].Count(), 2);
    AssertEqual("Line 0 word 0 correct", afb[0][0], "SIFFa");
    AssertEqual("Line 0 word 1 correct", afb[0][1], "1.0");
    AssertEqual<size_t>("Line 1 has 1 words", afb[1].Count(), 7);
    AssertEqual("Line 1 word 0 correct", afb[1][0], "0");
    AssertEqual<size_t>("Line 2 has 6 words", afb[2].Count(), 7);
    AssertEqual("Line 2 word 0 correct", afb[2][0], "1");
    AssertEqual<size_t>("Line 3 has 7 words", afb[3].Count(), 7);
    AssertEqual<size_t>("Line 4 has 7 words", afb[4].Count(), 7);
    AssertEqual<size_t>("Line 5 has 7 words", afb[5].Count(), 7);
    AssertEqual<size_t>("Line 6 has 7 words", afb[6].Count(), 7);
    AssertEqual<size_t>("Line 7 has 7 words", afb[7].Count(), 7);
    AssertEqual<size_t>("Line 8 has 7 words", afb[8].Count(), 7);
    AssertEqual<size_t>("Line 9 has 7 words", afb[9].Count(), 7);
#endif
}


/*
 * TestNamedColours
 */
void TestNamedColours(void) {
    using vislib::graphics::NamedColours;
    using vislib::graphics::ColourRGBAu8;
    using vislib::graphics::ColourParser;

    AssertEqual<size_t>("CountNamedColours == 149", NamedColours::CountNamedColours(), 149);
    for (size_t i = 0; i < NamedColours::CountNamedColours(); i++) {
        the::astring n = NamedColours::GetNameByIndex(i);
        const ColourRGBAu8& c = NamedColours::GetColourByIndex(i);
        const char *n2 = NamedColours::GetNameByColour(c, false);
        AssertNotEqual<const char*>("Named colour has name", n2, NULL);
        if (n2 == NULL) continue;
        // Do not test names, as there are some ambigious names (White = Transparent; Aqua = Cyan; ...)
        //if (    !n.Equals("Transparent")
        //        && !n.Equals("Fuchsia")
        //        && !n.Equals("Aqua")) {
        //    the::astring msg;
        //    the::text::astring_builder::format_to(msg, "Named colours name %s is correct", n.c_str());
        //    AssertTrue(msg.c_str(), n.Equals(n2, false));
        //}
        const ColourRGBAu8 c1 = NamedColours::GetColourByName(n.c_str());
        const ColourRGBAu8 c2 = NamedColours::GetColourByName(n2);
        AssertEqual("Colour[i] == Colour[Name[i]]", c, c1);
        AssertEqual("Colour[i] == Colour[Name[Colour[i]]]", c, c2);
    }

    AssertEqual("BurlyWood", NamedColours::BurlyWood, NamedColours::GetColourByName("BurlyWood"));
    AssertEqual("Firebrick", NamedColours::Firebrick, NamedColours::GetColourByName("Firebrick"));
    AssertEqual("HotPink", NamedColours::HotPink, NamedColours::GetColourByName("HotPink"));
    AssertEqual("PeachPuff", NamedColours::PeachPuff, NamedColours::GetColourByName("PeachPuff"));
    AssertEqual("PowderBlue", NamedColours::PowderBlue, NamedColours::GetColourByName("PowderBlue"));
    AssertEqual("Thistle", NamedColours::Thistle, NamedColours::GetColourByName("Thistle"));

    AssertNotEqual("BurlyWood != Firebrick", NamedColours::BurlyWood, NamedColours::Firebrick);
    AssertNotEqual("HotPink != PeachPuff", NamedColours::HotPink, NamedColours::PeachPuff);
    AssertNotEqual("PowderBlue != Thistle", NamedColours::PowderBlue, NamedColours::Thistle);
    AssertNotEqual("BurlyWood != HotPink", NamedColours::BurlyWood, NamedColours::HotPink);
    AssertNotEqual("BurlyWood != PowderBlue", NamedColours::BurlyWood, NamedColours::PowderBlue);
    AssertNotEqual("HotPink != PowderBlue", NamedColours::HotPink, NamedColours::PowderBlue);

    ColourRGBAu8 col;
    ColourParser::FromString("AliceBlue", col);
    AssertEqual("AliceBlue parsed correctly", col, NamedColours::AliceBlue);

    AssertNoException("ColourParser::FromString #1", ColourParser::FromString("(1,0, 0,5, 0,25)", col, true));
    AssertEqual("FromString #1 Colour correct", col, ColourRGBAu8(255, 128, 64, 255));

    AssertNoException("ColourParser::FromString #2", ColourParser::FromString("(1,0,0,5,0,25,1,0)", col, true));
    AssertEqual("FromString #2 Colour correct", col, ColourRGBAu8(255, 128, 64, 255));

    AssertNoException("ColourParser::FromString #3", ColourParser::FromString("(1,0, 0.5, 0,25, 1.0)", col, true));
    AssertEqual("FromString #3 Colour correct", col, ColourRGBAu8(255, 128, 64, 255));

    the::astring txt;
    ColourParser::ToString(txt, NamedColours::Crimson, ColourParser::REPTYPE_HTML);
    AssertTrue("Crimson in HTML correct", the::text::string_utility::compare(txt, "#dc143c", false) == 0);
    ColourParser::FromString(txt, col);
    AssertEqual("Crimson parsed back correctly from html", col, NamedColours::Crimson);

    ColourParser::ToString(txt, NamedColours::CornflowerBlue, ColourParser::REPTYPE_BYTE);
    AssertTrue("CornflowerBlue in Bytes correct", the::text::string_utility::compare(txt, "(100; 149; 237)", false) == 0);
    ColourParser::FromString(txt, col);
    AssertEqual("CornflowerBlue parsed back correctly from bytes", col, NamedColours::CornflowerBlue);

    ColourParser::ToString(txt, NamedColours::MediumVioletRed, ColourParser::REPTYPE_NAMED);
    AssertTrue("MediumVioletRed in Named correct", the::text::string_utility::compare(txt, "MediumVioletRed", false) == 0);
    ColourParser::FromString(txt, col);
    AssertEqual("MediumVioletRed parsed back correctly from name", col, NamedColours::MediumVioletRed);

    ColourParser::ToString(txt, NamedColours::SpringGreen, ColourParser::REPTYPE_FLOAT);
    float r, g, b;
    int cnt = 
#ifdef _WIN32
        sscanf_s
#else /* _WIN32 */
        sscanf
#endif /* _WIN32 */
        (txt.c_str(), "(%f; %f; %f)", &r, &g, &b);
    AssertTrue("SpringGreen in Floats correct", 
         (cnt == 3)
         && vislib::math::IsEqual(r, 0.0f)
         && vislib::math::IsEqual(g, 1.0f)
         && vislib::math::IsEqual(b, 0.49803901f) );
    ColourParser::FromString(txt, col);
    AssertEqual("SpringGreen parsed back correctly from floats", col, NamedColours::SpringGreen);

}
