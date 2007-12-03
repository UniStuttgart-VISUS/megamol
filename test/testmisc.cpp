/*
 * testmisc.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testmisc.h"
#include <vislib/NetworkInformation.h>
#include "testhelper.h"

#include <vislib/Console.h>
#include <vislib/ColumnFormatter.h>
#include <vislib/Exception.h>
#include <vislib/SystemException.h>
#include <vislib/PerformanceCounter.h>
#include "vislib/SingleLinkedList.h"
#include <vislib/StringConverter.h>
#include <vislib/SystemMessage.h>
#include <vislib/Path.h>
#include <vislib/Trace.h>

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

char *TestNIHelper1(vislib::net::NetworkInformation::Adapter::ValidityType v) {
    switch (v) {
        case vislib::net::NetworkInformation::Adapter::NOT_VALID : return "NOT_VALID"; break;
        case vislib::net::NetworkInformation::Adapter::VALID : return "VALID"; break;
        case vislib::net::NetworkInformation::Adapter::VALID_GENERATED : return "VALID_GEN"; break;
        default: return "UNKNOWN"; break;
    }
}

void TestNetworkInformation(void) {
    unsigned int cnt = vislib::net::NetworkInformation::AdapterCount();
    printf("%u Adapters found: \n", cnt);

    for (unsigned int i = 0; i < cnt; i++) {
        const vislib::net::NetworkInformation::Adapter &ad = vislib::net::NetworkInformation::AdapterInformation(i);
        printf("\tAdapter %u\n", i);
        printf("\tName[%s]: %s\n", TestNIHelper1(ad.NameValidity()), ad.Name().PeekBuffer());
        printf("\tMAC[%s]: %s\n", TestNIHelper1(ad.MACAddressValidity()), ad.MACAddress().PeekBuffer());
        printf("\tAddr[%s]: %s\n", TestNIHelper1(ad.AddressValidity()), ad.Address().ToStringA().PeekBuffer());
        printf("\tMask[%s]: %s\n", TestNIHelper1(ad.SubnetMaskValidity()), ad.SubnetMask().ToStringA().PeekBuffer());
        printf("\tBroadcast[%s]: %s\n", TestNIHelper1(ad.BroadcastAddressValidity()), ad.BroadcastAddress().ToStringA().PeekBuffer());

    }

}

void TestTrace(void) {
    vislib::Trace::GetInstance().EnableFileOutput("trace.txt");
    vislib::Trace::GetInstance().EnableDebuggerOutput(true);
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ALL);
    TRACE(1, "HORST!\n");
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


int intSortCompare(const int& lhs, const int& rhs) {
    return lhs - rhs;
}


void TestSingleLinkedListSort(void) {
    vislib::SingleLinkedList<int> list;

    list.Add(22);
    list.Add(8);
    list.Add(21);
    list.Add(22);
    list.Add(50);
    list.Add(2);
    list.Add(1);
    list.Add(10);

    AssertEqual<int>("List filled with 8 Elements", int(list.Count()), 8);

    list.Sort(intSortCompare);

    AssertEqual<int>("List still contains 8 Elements", int(list.Count()), 8);

    vislib::SingleLinkedList<int>::Iterator iter = list.GetIterator();
    AssertTrue("Iterator before Element 1", iter.HasNext());
    AssertEqual("Element 1 = 1", iter.Next(), 1);
    AssertTrue("Iterator before Element 2", iter.HasNext());
    AssertEqual("Element 2 = 2", iter.Next(), 2);
    AssertTrue("Iterator before Element 3", iter.HasNext());
    AssertEqual("Element 3 = 8", iter.Next(), 8);
    AssertTrue("Iterator before Element 4", iter.HasNext());
    AssertEqual("Element 4 = 10", iter.Next(), 10);
    AssertTrue("Iterator before Element 5", iter.HasNext());
    AssertEqual("Element 5 = 21", iter.Next(), 21);
    AssertTrue("Iterator before Element 6", iter.HasNext());
    AssertEqual("Element 6 = 22", iter.Next(), 22);
    AssertTrue("Iterator before Element 7", iter.HasNext());
    AssertEqual("Element 7 = 22", iter.Next(), 22);
    AssertTrue("Iterator before Element 8", iter.HasNext());
    AssertEqual("Element 8 = 50", iter.Next(), 50);
    AssertFalse("Iterator at end of list", iter.HasNext());

}
