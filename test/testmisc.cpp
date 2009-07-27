/*
 * testmisc.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testmisc.h"
#include "testhelper.h"

#include "vislib/Array.h"
#include "vislib/Console.h"
#include "vislib/ConsoleProgressBar.h"
#include "vislib/ColumnFormatter.h"
#include "vislib/Exception.h"
#include "vislib/SystemException.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemMessage.h"
#include "vislib/Path.h"
#include "vislib/Trace.h"
#include "vislib/FileNameSequence.h"
#include "vislib/BitmapImage.h"
#include "vislib/PpmBitmapCodec.h"
#include "vislib/RawStorage.h"

// #define USE_UNICODE_COLUMNFORMATTER

#ifndef USE_UNICODE_COLUMNFORMATTER
typedef vislib::ColumnFormatterA ColumnFormatter;
typedef vislib::StringA CFString;
typedef char CFChar;
#define CFS(A) A

#else // USE_UNICODE_COLUMNFORMATTER
typedef vislib::ColumnFormatterW ColumnFormatter;
typedef vislib::StringW CFString;
typedef wchar_t CFChar;
#define CFS(A) L ## A

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
    std::cout << expected.PeekBuffer() << std::endl;
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    std::cout << "Result:" << std::endl;
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << result.PeekBuffer() << std::endl;
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

    AssertException("Accessing Column[3]: InvalidParamException", ColFormatter[3].SetWidth(0), vislib::IllegalParamException);

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
               CFS("       | comes with some word |                          | very easy.\n")
               CFS("       | wraps.               |                          |");
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
               CFS("       | some out of column   | test!                    | test 4.\n")
               CFS("       | tests.               |                          |");
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
               CFS("very long first column as hardcore test. | The 3. Column | This was the fourth\n")
               CFS("       | 2. Column is in 3.   | starts where it should   | test.\n")
               CFS("       | line                 | be.                      |");
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
               CFS("                    .--.                              .--.123456789");
    if (!AssertEqual("Formatted 5. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[0].SetWidth(102);

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
               CFS("                    .--.                              .--.123456789");
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
               CFS("                    .--.56789                         .--.012345678901234567890\n")
               CFS("                    .--.                              .--.123456789");
    if (!AssertEqual("Formatted 7. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].EnableWrapping();

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
    expected = CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
               CFS("                    .--.012345678901234567890123456789.--.123456789");
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
               CFS("                    .--.Horst                         .--.123456789");
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
               CFS("                    .--.push into a new line..--.Third Column");
    if (!AssertEqual("Formatted 11. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column is now shorter, But still too long"));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.The second column is now shorter, But still too long\n")
               CFS("                    .--..--.Third Column");
    if (!AssertEqual("Formatted 12. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column is now a bit longer again.  I want to check space interpretation on word wrap here."));

    ColFormatter >> output;
//                  1234567890123456789012345678901234567890123456789012345678901234567890123456789
//                  01234567890123456789.--.01234.--.012345678901234567890123456789
    expected = CFS("First Column        .--.The second column is now a bit longer again.  I want to\n")
               CFS("                    .--.check space interpretation on word wrap here..--.Third\n")
               CFS("                    .--.                                             .--.Column");
    if (!AssertEqual("Formatted 13. output as expected", output, expected)) {
        TestColumnFormatterHelper(output, expected);
    }

}


void TestTrace(void) {
    vislib::Trace::GetInstance().EnableFileOutput("trace.txt");
    vislib::Trace::GetInstance().EnableDebuggerOutput(true);
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ALL);
    VLTRACE(1, "HORST!\n");
}

void TestExceptions(void) {
    vislib::sys::SystemException e1(2, __FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e1.GetMsg());

    vislib::Exception e2(__FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e2.GetMsg()); 
}

void TestSystemMessage(void) {
    vislib::sys::SystemMessage sysMsg(4);
    ::_tprintf(_T("%s\n"), static_cast<const TCHAR *>(sysMsg));
}

void TestPerformanceCounter(void) {
    for (int i = 0; i < 100; i++) {
        ::_tprintf(_T("%lu\n"), (unsigned long)vislib::sys::PerformanceCounter::Query());
    }
}

void TestPathManipulations(void) {
    // Hazard! Think of some more secure test cases.

    //try {
    //    vislib::sys::Path::MakeDirectory(L"Horst/Hugo/Heinz/Hans/Helmut");
    //    vislib::sys::Path::DeleteDirectory("Wurst", true);
    //} catch(vislib::sys::SystemException e) {
    //    fprintf(stderr, "SystemException: %s\n", e.GetMsgA());
    //} catch(vislib::Exception e) {
    //    fprintf(stderr, "Exception: %s\n", e.GetMsgA());
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

    AssertEqual("File 0 is correct", seq.FileNameA(0), "C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 0 is correct", seq.FileNameW(0), L"C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 1 is correct", seq.FileNameA(1), "C:\\Some\\Directory\\AndFileName01_0003.dmy");
    AssertEqual("File 2 is correct", seq.FileNameA(2), "C:\\Some\\Directory\\AndFileName01_0006.dmy");
    AssertEqual("File 3 is correct", seq.FileNameA(3), "C:\\Some\\Directory\\AndFileName01_0009.dmy");
    AssertEqual("File 4 is correct", seq.FileNameA(4), "C:\\Some\\Directory\\AndFileName02_0000.dmy");
    AssertEqual("File 5 is correct", seq.FileNameA(5), "C:\\Some\\Directory\\AndFileName02_0003.dmy");
    AssertEqual("File 6 is correct", seq.FileNameA(6), "C:\\Some\\Directory\\AndFileName02_0006.dmy");
    AssertEqual("File 7 is correct", seq.FileNameA(7), "C:\\Some\\Directory\\AndFileName02_0009.dmy");

    seq.SetReversedCounterPriority(true);
    AssertTrue("Sequence is in reversed order", seq.ReversedCounterPriority());

    AssertEqual("File 0 is correct", seq.FileNameA(0), "C:\\Some\\Directory\\AndFileName01_0000.dmy");
    AssertEqual("File 1 is correct", seq.FileNameA(1), "C:\\Some\\Directory\\AndFileName02_0000.dmy");
    AssertEqual("File 2 is correct", seq.FileNameA(2), "C:\\Some\\Directory\\AndFileName01_0003.dmy");
    AssertEqual("File 3 is correct", seq.FileNameA(3), "C:\\Some\\Directory\\AndFileName02_0003.dmy");
    AssertEqual("File 4 is correct", seq.FileNameA(4), "C:\\Some\\Directory\\AndFileName01_0006.dmy");
    AssertEqual("File 5 is correct", seq.FileNameA(5), "C:\\Some\\Directory\\AndFileName02_0006.dmy");
    AssertEqual("File 6 is correct", seq.FileNameA(6), "C:\\Some\\Directory\\AndFileName01_0009.dmy");
    AssertEqual("File 7 is correct", seq.FileNameA(7), "C:\\Some\\Directory\\AndFileName02_0009.dmy");

    // no need to clean up s1, s2, s3, c1, and c2 because they have been
    // assigned to smart pointers ;-)

    /*
    seq.Autodetect("T:\\grottel\\Photos\\HotelRoom1\\DSC00004.JPG");
    if (seq.IsValid()) {
        unsigned int cnt = seq.Count();
        printf("Autodetected %u files:\n", cnt);
        for (unsigned int i = 0; i < cnt; i++) {
            printf("  %s\n", seq.FileNameA(i).PeekBuffer());
        }
    } else {
        printf("Autodetection failed (Sequence is invalid).\n");
    }
    //*/
}


/*
 * TestBitmapCodecSimple
 */
void TestBitmapCodecSimple(void) {
    using vislib::graphics::PpmBitmapCodec;
    using vislib::graphics::BitmapImage;
    using vislib::RawStorage;

    const unsigned char bmpdata1[] = {
          0,  0,  0,  85,  0,  0, 170,  0,  0, 255,  0,  0,
          0, 85,  0,  85, 85,  0, 170, 85,  0, 255, 85,  0,
          0,170,  0,  85,170,  0, 170,170,  0, 255,170,  0,
          0,255,  0,  85,255,  0, 170,255,  0, 255,255,  0
    };

    const unsigned char bmpdata2[] = {
          0,  0,  0,  85,  0,  0, 170,  0,  0, 255,  0,  0,
          0,  0, 85,  85,  0, 85, 170,  0, 85, 255,  0, 85,
          0,  0,170,  85,  0,170, 170,  0,170, 255,  0,170,
          0,  0,255,  85,  0,255, 170,  0,255, 255,  0,255
    };

    PpmBitmapCodec codec;
    BitmapImage img(4, 4, 3, BitmapImage::CHANNELTYPE_BYTE, NULL);
    RawStorage mem;
    img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
    img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
    img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
    memcpy(img.PeekData(), bmpdata1, 4 * 4 *3);
    codec.Image() = &img;

    codec.SetSaveOption(true);
    AssertTrue("Working with binary PPM", codec.GetSaveOption());
    AssertTrue("Codec can store to memory", codec.CanSaveToMemory());
    AssertTrue("Codec can load from memory", codec.CanLoadFromMemory());
    AssertTrue("Bitmap data stored in image", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    AssertTrue("Image stored in memory", codec.Save(mem));
    AssertTrue("Memory not empty", mem.GetSize() > 0);
    AssertTrue("Image data not changed", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    memcpy(img.PeekData(), bmpdata2, 4 * 4 *3);
    AssertTrue("Image data updated (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) != 0);
    AssertTrue("Image data updated (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) == 0);

    AssertTrue("Image loaded from memory", codec.Load(mem));
    AssertTrue("Memory not cleared", mem.GetSize() > 0);
    AssertTrue("Image data restored (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);
    AssertTrue("Image data restored (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) != 0);

    codec.SetSaveOption(false);
    AssertFalse("Working with ASCII PPM", codec.GetSaveOption());
    AssertTrue("Codec can store to memory", codec.CanSaveToMemory());
    AssertTrue("Codec can load from memory", codec.CanLoadFromMemory());
    AssertTrue("Bitmap data stored in image", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    AssertTrue("Image stored in memory", codec.Save(mem));
    AssertTrue("Memory not empty", mem.GetSize() > 0);
    AssertTrue("Image data not changed", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    memcpy(img.PeekData(), bmpdata2, 4 * 4 *3);
    AssertTrue("Image data updated (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) != 0);
    AssertTrue("Image data updated (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) == 0);

    AssertTrue("Image loaded from memory", codec.Load(mem));
    AssertTrue("Memory not cleared", mem.GetSize() > 0);
    AssertTrue("Image data restored (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);
    AssertTrue("Image data restored (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) != 0);

    /* vislib::sys::File f;
    if (f.Open("C:\\temp\\test.ppm", vislib::sys::File::WRITE_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
        f.Write(mem, mem.GetSize());
        f.Close();
    } */

}
