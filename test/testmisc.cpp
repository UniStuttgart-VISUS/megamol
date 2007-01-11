/*
 * testmisc.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testmisc.h"
#include "testhelper.h"

#include <vislib/Console.h>
#include <vislib/ColumnFormatter.h>

#define USE_UNICODE_COLUMNFORMATTER

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

void TestConsoleColors(void) {
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

void TestColumnFormatter(void) {
    ColumnFormatter ColFormatter;
    CFString output = CFS("Test");

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

    ColFormatter[0].SetText(CFS("1."));
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
    AssertEqual("Formatted 1. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
        CFS("1.     | This is 1. test      | All Text fits into line  | No wrapping at all"));

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
    AssertEqual("Formatted 2. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
        CFS("Second | The second test      | But no out of columns    | So the wrapping is  \n")
        CFS("       | comes with some word |                          | very easy.          \n")
        CFS("       | wraps.               |                          | "));

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
    AssertEqual("Formatted 3. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
        CFS("The Third Test | TTT performs | A short out of column    | Seeing forward to   \n")
        CFS("       | some out of column   | test!                    | test 4.             \n")
        CFS("       | tests.               |                          | "));

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
    AssertEqual("Formatted 4. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           123456 | 12345678901234567890 | 123456789012345678901234 | 12345678901234567890
        CFS("The fourth test comes with a very very very very very very very very very very \n")
        CFS("very long first column as hardcore test. | The 3. Column | This was the fourth \n")
        CFS("       | 2. Column is in 3.   | starts where it should   | test.               \n")
        CFS("       | line                 | be.                      | "));

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
    AssertEqual("Formatted 5. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
        CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
        CFS("                    .--.                              .--.123456789"));

//----------------------------------------------------------------------------
    ColFormatter[0].SetWidth(102);

    ColFormatter >> output;
    AssertEqual("Formatted 6. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
        CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
        CFS("                    .--.                              .--.123456789"));

//----------------------------------------------------------------------------
    ColFormatter[1].DisableWrapping();
    ColFormatter[1].SetText(CFS("012345678901234567890123456789012345678901234567890123456789"));

    ColFormatter >> output;
    AssertEqual("Formatted 7. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
        CFS("01234567890123456789.--.0123456789012345678901234567890123456789012345678901234\n")
        CFS("                    .--.56789                         .--.012345678901234567890\n")
        CFS("                    .--.                              .--.123456789"));

//----------------------------------------------------------------------------
    ColFormatter[1].EnableWrapping();

    ColFormatter >> output;
    AssertEqual("Formatted 8. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
        CFS("01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890\n")
        CFS("                    .--.012345678901234567890123456789.--.123456789"));                         

//----------------------------------------------------------------------------
    ColFormatter[0].SetWidth(20);
    ColFormatter[0].DisableWrapping();
    ColFormatter[0].SetText(CFS("012345678901234567890123456789012345678901234567890"));
    ColFormatter[1].SetText(CFS("Horst"));

    ColFormatter >> output;
    AssertEqual("Formatted 9. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.012345678901234567890123456789.--.012345678901234567890123456789
        CFS("012345678901234567890123456789012345678901234567890   .--.012345678901234567890\n")
        CFS("                    .--.Horst                         .--.123456789"));                         

//----------------------------------------------------------------------------
    ColFormatter[1].SetWidth(0);
    ColFormatter[0].SetText(CFS("First Column"));
    ColFormatter[2].SetText(CFS("Third Column"));

    ColFormatter >> output;
    AssertEqual("Formatted 10. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.01234.--.012345678901234567890123456789
        CFS("First Column        .--.Horst.--.Third Column"));

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column now gets a very very long text to push into a new line."));

    ColFormatter >> output;
    AssertEqual("Formatted 11. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.01234.--.012345678901234567890123456789
        CFS("First Column        .--.The second column now gets a very very long text to    \n")
        CFS("                    .--.push into a new line..--.Third Column"));

//----------------------------------------------------------------------------
    ColFormatter[1].SetText(CFS("The second column is now shorter, But still too long"));

    ColFormatter >> output;
    AssertEqual("Formatted 12. output as expected", output,
//           1234567890123456789012345678901234567890123456789012345678901234567890123456789
//           01234567890123456789.--.01234.--.012345678901234567890123456789
        CFS("First Column        .--.The second column is now shorter, But still too long   \n")
        CFS("                    .--..--.Third Column"));

}
