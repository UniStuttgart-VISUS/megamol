/*
 * teststring.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "teststring.h"
#include "testhelper.h"

#include "vislib/String.h"
#include "vislib/CharTraits.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"


void TestString(void) {
    using namespace vislib;

    //::printf("%s", W2A(L"Hugo"));

    TestStringA();
    TestStringW();

    StringA a1("Horst");
    StringW w1(a1);
    StringW w2(L"Hugo");
    StringA a2(w2);

    AssertTrue("ANSI to wide character conversion constructor", !::wcscmp(w1, L"Horst"));
    AssertTrue("Wide to ANSI character conversion constructor", !::strcmp(a2, "Hugo"));

    a1 = StringW(L"Hällo Wörld");
    w1 = StringA("Hällo Wörld");
    AssertTrue("assignment-conversion-test #3", !::strcmp(a1, "Hällo Wörld"));
    AssertTrue("assignment-conversion-test #4", !::wcscmp(w1, L"Hällo Wörld"));

    //wchar_t wc = L'\x00e4';
    //char ac = 0;
    //CharTraitsW::Convert(ac, wc);
    //AssertEqual("Wide to ANSI character conversion.", ac, char('ä'));

    //ac = char('ö');
    //CharTraitsA::Convert(wc, ac);
    //AssertEqual("ANSI to wide character conversion.", wc, L'\x00f6');

    //TestUTF8String();

    TestStringTokeniser();

}


void TestStringA(void) {
    using namespace vislib;

    StringA s1;
    StringA s2("Horst");
    StringA s3(s2);
    StringA s4('h', 5);

    AssertTrue("Default Constructor creates empty string", !::strcmp(s1.PeekBuffer(), ""));
    AssertTrue("C string constructor", !::strcmp(s2.PeekBuffer(), "Horst"));
    AssertTrue("Copy constructor", !::strcmp(s2.PeekBuffer(), s3.PeekBuffer()));
    AssertTrue("Character constructor", !::strcmp(s4.PeekBuffer(), "hhhhh"));

    AssertTrue("Test for empty string", s1.IsEmpty());
    AssertFalse("Test for emtpy string", s2.IsEmpty());

    AssertEqual("\"Horst\" consists of 5 characters", s2.Length(), 5);

    AssertEqual("\"Horst\"[0] is 'H'", s2[0], 'H');
    AssertEqual("\"Horst\"[4] is 't'", s2[4], 't');

    try {
        s2[-1];
        AssertTrue("OutOfRangeException at begin", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at begin", true);
    }

    try {
        s2[s2.Length()];
        AssertTrue("OutOfRangeException at end", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at end", true);
    }

    AssertTrue("Test for inequality", s1 != s2);
    AssertTrue("Test for equality", s2 == s3);

    s1 = s2;
    AssertTrue("Assignment operator", !::strcmp(s1.PeekBuffer(), s2.PeekBuffer()));

    AssertTrue("\"Horst\" begins with \"H\"", s1.StartsWith("H"));
    AssertTrue("\"Horst\" begins with \"Ho\"", s1.StartsWith("Ho"));
    AssertTrue("\"Horst\" begins with \"Horst\"", s1.StartsWith("Horst"));
    AssertFalse("\"Horst\" does not begin with \"Hu\"", s1.StartsWith("Hu"));
    AssertTrue("\"Horst\" begins with 'H'", s1.StartsWith('H'));
    AssertTrue("\"Horst\" begins (insensitive) with 'h'", s1.StartsWith('h', false));
    AssertTrue("\"Horst\" begins (insensitive) with 'horst'", s1.StartsWith('h', false));
    AssertFalse("\"Horst\" does not begin (insensitive) with \"hu\"", s1.StartsWith("hu", false));

    AssertTrue("\"Horst\" ends with \"t\"", s1.EndsWith("t"));
    AssertTrue("\"Horst\" ends with \"st\"", s1.EndsWith("st"));
    AssertFalse("\"Horst\" does not ends with \"go\"", s1.EndsWith("go"));
    AssertTrue("\"Horst\" ends with 't'", s1.EndsWith('t'));
    AssertTrue("\"Horst\" ends with 't'", s1.EndsWith("Horst"));

    AssertEqual("First 'o' in \"Horst\"", s1.Find('o'), 1);
    AssertEqual("First 'o' in \"Horst\" after 2", s1.Find('o', 2), StringA::INVALID_POS);
    AssertEqual("Last 'o' in \"Horst\"", s1.FindLast('o'), 1);
    AssertEqual("Last 't' in \"Horst\"", s1.FindLast('t'), 4);
    AssertEqual("First \"ors\" in \"Horst\"", s1.Find("ors"), 1);
    AssertEqual("First \"ors\" in \"Horst\" after 2", s1.Find("ors", 2), StringA::INVALID_POS);
    AssertEqual("First \"rs\" in \"Horst\" after 2", s1.Find("rs", 2), 2);
    AssertEqual("First \"ort\" in \"Horst\"", s1.Find("ort"), StringA::INVALID_POS);
    AssertEqual("Last \"ors\" in \"Horst\"", s1.FindLast("ors"), 1);
    AssertEqual("Last \"orst\" in \"Horst\"", s1.FindLast("orst"), 1);
    AssertEqual("Last \"orst2\" in \"Horst\"", s1.FindLast("orst2"), StringA::INVALID_POS);
    AssertEqual("Last \"orst2\" in \"Horst\"", s1.FindLast("orst2"), StringA::INVALID_POS);
    AssertEqual("Last \"rst\" in \"Horst\"", s1.FindLast("rst"), 2);
    AssertEqual("Last \"rst\" in \"Horst\" before 1", s1.FindLast("rst", 1), StringA::INVALID_POS);
    AssertEqual("Last \"or\" in \"Horst\" before 2", s1.FindLast("or", 2), 1);

    s4 = s2 + " und Hugo";
    AssertTrue("Concatenation", !::strcmp(s4.PeekBuffer(), "Horst und Hugo"));

    s2 += " und Hugo";
    AssertTrue("Assignment concatenation", !::strcmp(s2.PeekBuffer(), "Horst und Hugo"));

    s2.Replace('H', 'h');
    AssertTrue("Character replacement", !::strcmp(s2.PeekBuffer(), "horst und hugo"));

    s2 = "Horst";
    s2.Append(" und Hugo");
    AssertTrue("Append", !::strcmp(s2.PeekBuffer(), "Horst und Hugo"));
    s2.Prepend("Heinz und ");
    AssertTrue("Prepend", !::strcmp(s2.PeekBuffer(), "Heinz und Horst und Hugo"));

    s1.Format("Horst");
    AssertTrue("Format \"Horst\"", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1.Format("Horst%d", 2);
    AssertTrue("Format \"Horst2\"", !::strcmp(s1.PeekBuffer(), "Horst2"));

    s1.Format("Horst%d %04.1f", 2, 2.0f);
    AssertTrue("Format \"Horst2 02.0\"", !::strcmp(s1.PeekBuffer(), "Horst2 02.0"));

    s1.Format("Horst %S", L"Hugo");
    AssertTrue("Format \"Horst Hugo\"", !::strcmp(s1.PeekBuffer(), "Horst Hugo"));

    s1 = "Horst";
    s1.ToLowerCase();
    AssertTrue("ToLowerCase", !::strcmp(s1.PeekBuffer(), "horst"));
    s1.ToUpperCase();
    AssertTrue("ToUpperCase", !::strcmp(s1.PeekBuffer(), "HORST"));

    s1 = "Horst";
    s1.TrimBegin("oH");
    AssertTrue("\"Horst\" trim begin \"oH\"", !::strcmp(s1.PeekBuffer(), "rst"));

    s1 = "Horst";
    s1.TrimEnd("st");
    AssertTrue("\"Horst\" trim end \"st\"", !::strcmp(s1.PeekBuffer(), "Hor"));

    s1 = "Horst";
    s1.TrimEnd("rt");
    AssertTrue("\"Horst\" trim end \"rt\"", !::strcmp(s1.PeekBuffer(), "Hors"));

    s1 = "Horst";
    s2 = s1.Substring(1);
    AssertTrue("\"Horst\".Substring(1)", !::strcmp(s2.PeekBuffer(), "orst"));

    s1 = "Horst";
    s2 = s1.Substring(10);
    AssertTrue("\"Horst\".Substring(10)", !::strcmp(s2.PeekBuffer(), ""));

    s1 = "Horst";
    s2 = s1.Substring(1, 3);
    AssertTrue("\"Horst\".Substring(1, 3)", !::strcmp(s2.PeekBuffer(), "ors"));

    s1 = "Horst";
    s2 = s1.Substring(10, 3);
    AssertTrue("\"Horst\".Substring(10, 3)", !::strcmp(s2.PeekBuffer(), ""));

    s1 = "Horst";
    s2 = s1.Substring(1, 5);
    AssertTrue("\"Horst\".Substring(1, 5)", !::strcmp(s2.PeekBuffer(), "orst"));

    s1 = "Hors";
    s2 = s1 + 't';
    AssertTrue("Character concatenation", !::strcmp(s2.PeekBuffer(), "Horst"));

    s1 = "Hors";
    s1 += 't';
    AssertTrue("Character assignment concatenation", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Hors";
    s1.Append('t');
    AssertTrue("Character append", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "orst";
    s1.Prepend('H');
    AssertTrue("Character prepend", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Horst Horst Horst";
    AssertEqual("3 replacements.", s1.Replace("Horst", "Hugo"), 3);
    AssertTrue("String replacement", !::strcmp(s1.PeekBuffer(), "Hugo Hugo Hugo"));
    AssertEqual("0 replacements.", s1.Replace("Horst", "Hugo"), 0);
    
    s1 = "Horst und Hugo";
    AssertEqual("1 replacement.", s1.Replace("und", "oder"), 1);
    AssertTrue("String replacement", !::strcmp(s1.PeekBuffer(), "Horst oder Hugo"));

    s1 = "Horst und Hugo";
    s1.Remove(5);
    AssertTrue("Remove(5)", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Horst und Hugo";
    s1.Remove(5, 4);
    AssertTrue("Remove(5, 4)", !::strcmp(s1.PeekBuffer(), "Horst Hugo"));

    s1 = "Horst und Hugo";
    s1.Remove(5, 9);
    AssertTrue("Remove(5, 9)", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Horst und Hugo";
    s1.Remove(5, 8);
    AssertTrue("Remove(5, 8)", !::strcmp(s1.PeekBuffer(), "Horsto"));

    s1 = "Horst und Hugo";
    s1.Remove("und ");
    AssertTrue("Remove \"und \"", !::strcmp(s1.PeekBuffer(), "Horst Hugo"));

    s1 = "Horst";
    s1.Clear();
    AssertTrue("Cleared string is emtpy", s1.IsEmpty());

#ifdef _WIN32
    AssertTrue("Resource found.", s1.Load(101));
    AssertTrue("Load string from resource", !::strcmp(s1.PeekBuffer(), "Das ist ein Test."));
#endif /* _WIN32 */

    s1 = "Horst Horst";
    s1.Replace("orst", "ugo", 1);
    AssertTrue("Limited string replace", !::strcmp(s1.PeekBuffer(), "Hugo Horst"));

    s1 = "Horst Horst";
    s1.Replace("orst", "ugo", StringA::NO_LIMIT);
    AssertTrue("Unlimited string replace", !::strcmp(s1.PeekBuffer(), "Hugo Hugo"));

    s1 = "Horst Horst";
    s1.Replace('o', 'i', 1);
    AssertTrue("Limited char replace", !::strcmp(s1.PeekBuffer(), "Hirst Horst"));

    s1 = "Horst Horst";
    s1.Replace('o', 'i', StringA::NO_LIMIT);
    AssertTrue("Unlimited char replace", !::strcmp(s1.PeekBuffer(), "Hirst Hirst"));

    s1 = "Hrst";
    s1.Insert(1, 'o');
    AssertTrue("Character insert at 1", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "orst";
    s1.Insert(0, 'H');
    AssertTrue("Character insert at 0", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Hors";
    s1.Insert(4, 't');
    AssertTrue("Character insert at end", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Ht";
    s1.Insert(1, "ors");
    AssertTrue("String insert at 1", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "rst";
    s1.Insert(0, "Ho");
    AssertTrue("String insert at 0", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Hor";
    s1.Insert(3, "st");
    AssertTrue("String insert at end", !::strcmp(s1.PeekBuffer(), "Horst"));

    s1 = "Hugo %d";
    s2.Format(&s1, 3);
    AssertTrue("Format with vislib format string", !::strcmp(s2.PeekBuffer(), "Hugo 3"));

	s1 = "The quick brown fox jumps over the lazy dog.";
	s2 = s1;
	s2.Rot48();
	AssertNotEqual("Obfuscated is different", s1, s2);
	s2.Rot48();
	AssertEqual("Deobfuscated is equal to original", s1, s2);

    s1 = "Tier";
    s2 = "Tor";
    AssertEqual("Levenshtein distance.", s1.LevenshteinDistance(s2), StringA::Size(2));

    s1 = "kitten";
    s2 = "sitting";
    AssertEqual("Levenshtein distance.", s1.LevenshteinDistance(s2), StringA::Size(3));
}

void TestStringW(void) {
    using namespace vislib;

    StringW s1;
    StringW s2(L"Horst");
    StringW s3(s2);
    StringW s4(L'h', 5);

    AssertTrue("Default Constructor creates empty string", !::wcscmp(s1.PeekBuffer(), L""));
    AssertTrue("C string constructor", !::wcscmp(s2.PeekBuffer(), L"Horst"));
    AssertTrue("Copy constructor", !::wcscmp(s2.PeekBuffer(), s3.PeekBuffer()));
    AssertTrue("Character constructor", !::wcscmp(s4.PeekBuffer(), L"hhhhh"));

    AssertTrue("Test for empty string", s1.IsEmpty());
    AssertFalse("Test for emtpy string", s2.IsEmpty());

    AssertEqual("\"Horst\" consists of 5 characters", s2.Length(), 5);

    AssertEqual("\"Horst\"[0] is 'H'", s2[0], L'H');
    AssertEqual("\"Horst\"[4] is 't'", s2[4], L't');

    try {
        s2[-1];
        AssertTrue("OutOfRangeException at begin", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at begin", true);
    }

    try {
        s2[s2.Length()];
        AssertTrue("OutOfRangeException at end", false);
    } catch (OutOfRangeException) {
        AssertTrue("OutOfRangeException at end", true);
    }

    AssertTrue("Test for inequality", s1 != s2);
    AssertTrue("Test for equality", s2 == s3);

    s1 = s2;
    AssertTrue("Assignment operator", !::wcscmp(s1.PeekBuffer(), s2.PeekBuffer()));

    AssertTrue("\"Horst\" begins with \"H\"", s1.StartsWith(L"H"));
    AssertTrue("\"Horst\" begins with \"Ho\"", s1.StartsWith(L"Ho"));
    AssertTrue("\"Horst\" begins with \"Horst\"", s1.StartsWith(L"Horst"));
    AssertFalse("\"Horst\" does not begin with \"Hu\"", s1.StartsWith(L"Hu"));
    AssertTrue("\"Horst\" begins with 'H'", s1.StartsWith(L'H'));
    AssertTrue("\"Horst\" begins (insensitive) with 'h'", s1.StartsWith(L'h', false));
    AssertTrue("\"Horst\" begins (insensitive) with 'horst'", s1.StartsWith(L'h', false));
    AssertFalse("\"Horst\" does not begin (insensitive) with \"hu\"", s1.StartsWith(L"hu", false));

    AssertTrue("\"Horst\" ends with \"t\"", s1.EndsWith(L"t"));
    AssertTrue("\"Horst\" ends with \"st\"", s1.EndsWith(L"st"));
    AssertFalse("\"Horst\" does not ends with \"go\"", s1.EndsWith(L"go"));
    AssertTrue("\"Horst\" ends with 't'", s1.EndsWith(L't'));
    AssertTrue("\"Horst\" ends with 't'", s1.EndsWith(L"Horst"));

    AssertEqual("First 'o' in \"Horst\"", s1.Find(L'o'), 1);
    AssertEqual("First 'o' in \"Horst\" after 2", s1.Find(L'o', 2), StringA::INVALID_POS);
    AssertEqual("Last 'o' in \"Horst\"", s1.FindLast(L'o'), 1);
    AssertEqual("First \"ors\" in \"Horst\"", s1.Find(L"ors"), 1);
    AssertEqual("First \"ors\" in \"Horst\" after 2", s1.Find(L"ors", 2), StringA::INVALID_POS);
    AssertEqual("First \"rs\" in \"Horst\" after 2", s1.Find(L"rs", 2), 2);
    AssertEqual("First \"ort\" in \"Horst\"", s1.Find(L"ort"), StringA::INVALID_POS);
    AssertEqual("Last \"ors\" in \"Horst\"", s1.FindLast(L"ors"), 1);
    AssertEqual("Last \"orst\" in \"Horst\"", s1.FindLast(L"orst"), 1);
    AssertEqual("Last \"orst2\" in \"Horst\"", s1.FindLast(L"orst2"), StringA::INVALID_POS);
    AssertEqual("Last \"rst\" in \"Horst\" before 1", s1.FindLast(L"rst", 1), StringA::INVALID_POS);
    AssertEqual("Last \"or\" in \"Horst\" before 2", s1.FindLast(L"or", 2), 1);

    s4 = s2 + L" und Hugo";
    AssertTrue("Concatenation", !::wcscmp(s4.PeekBuffer(), L"Horst und Hugo"));

    s2 += L" und Hugo";
    AssertTrue("Assignment concatenation", !::wcscmp(s2.PeekBuffer(), L"Horst und Hugo"));

    s2.Replace('H', 'h');
    AssertTrue("Character replacement", !::wcscmp(s2.PeekBuffer(), L"horst und hugo"));

    s2 = L"Horst";
    s2.Append(L" und Hugo");
    AssertTrue("Append", !::wcscmp(s2.PeekBuffer(), L"Horst und Hugo"));
    s2.Prepend(L"Heinz und ");
    AssertTrue("Prepend", !::wcscmp(s2.PeekBuffer(), L"Heinz und Horst und Hugo"));

    s1.Format(L"Horst");
    AssertTrue("Format \"Horst\"", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1.Format(L"Horst%d", 2);
    AssertTrue("Format \"Horst2\"", !::wcscmp(s1.PeekBuffer(), L"Horst2"));

    s1.Format(L"Horst%d %04.1f", 2, 2.0f);
    AssertTrue("Format \"Horst2 02.0\"", !::wcscmp(s1.PeekBuffer(), L"Horst2 02.0"));

    s1.Format(L"Horst %hs", "Hugo");
    AssertTrue("Format \"Horst Hugo\"", !::wcscmp(s1.PeekBuffer(), L"Horst Hugo"));

    s1 = L"Horst";
    s1.ToLowerCase();
    AssertTrue("ToLowerCase", !::wcscmp(s1.PeekBuffer(), L"horst"));
    s1.ToUpperCase();
    AssertTrue("ToUpperCase", !::wcscmp(s1.PeekBuffer(), L"HORST"));

    s1 = L"Horst";
    s1.TrimBegin(L"oH");
    AssertTrue("\"Horst\" trim begin \"oH\"", !::wcscmp(s1.PeekBuffer(), L"rst"));

    s1 = L"Horst";
    s1.TrimEnd(L"st");
    AssertTrue("\"Horst\" trim end \"st\"", !::wcscmp(s1.PeekBuffer(), L"Hor"));

    s1 = L"Horst";
    s1.TrimEnd(L"rt");
    AssertTrue("\"Horst\" trim end \"rt\"", !::wcscmp(s1.PeekBuffer(), L"Hors"));

    s1 = L"Horst";
    s1.TrimEnd(L"st");
    AssertTrue("\"Horst\" trim end \"st\"", !::wcscmp(s1.PeekBuffer(), L"Hor"));

    s1 = L"Horst";
    s1.TrimEnd(L"rt");
    AssertTrue("\"Horst\" trim end \"rt\"", !::wcscmp(s1.PeekBuffer(), L"Hors"));

    s1 = L"Horst";
    s2 = s1.Substring(1);
    AssertTrue("\"Horst\".Substring(1)", !::wcscmp(s2.PeekBuffer(), L"orst"));

    s1 = L"Horst";
    s2 = s1.Substring(10);
    AssertTrue("\"Horst\".Substring(10)", !::wcscmp(s2.PeekBuffer(), L""));

    s1 = L"Horst";
    s2 = s1.Substring(1, 3);
    AssertTrue("\"Horst\".Substring(1, 3)", !::wcscmp(s2.PeekBuffer(), L"ors"));

    s1 = L"Horst";
    s2 = s1.Substring(10, 3);
    AssertTrue("\"Horst\".Substring(10, 3)", !::wcscmp(s2.PeekBuffer(), L""));

    s1 = L"Horst";
    s2 = s1.Substring(1, 5);
    AssertTrue("\"Horst\".Substring(1, 5)", !::wcscmp(s2.PeekBuffer(), L"orst"));

    s1 = L"Hors";
    s2 = s1 + L't';
    AssertTrue("Character concatenation", !::wcscmp(s2.PeekBuffer(), L"Horst"));

    s1 = L"Hors";
    s1 += L't';
    AssertTrue("Character assignment concatenation", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Hors";
    s1.Append(L't');
    AssertTrue("Character append", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"orst";
    s1.Prepend(L'H');
    AssertTrue("Character prepend", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Horst Horst Horst";
    AssertEqual("3 replacements.", s1.Replace(L"Horst", L"Hugo"), 3);
    AssertTrue("String replacement", !::wcscmp(s1.PeekBuffer(), L"Hugo Hugo Hugo"));
    AssertEqual("0 replacements.", s1.Replace(L"Horst", L"Hugo"), 0);
    
    s1 = L"Horst und Hugo";
    AssertEqual("1 replacement.", s1.Replace(L"und", L"oder"), 1);
    AssertTrue("String replacement", !::wcscmp(s1.PeekBuffer(), L"Horst oder Hugo"));

    s1 = L"Horst und Hugo";
    s1.Remove(5);
    AssertTrue("Remove(5)", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Horst und Hugo";
    s1.Remove(5, 4);
    AssertTrue("Remove(5, 4)", !::wcscmp(s1.PeekBuffer(), L"Horst Hugo"));

    s1 = L"Horst und Hugo";
    s1.Remove(5, 9);
    AssertTrue("Remove(5, 9)", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Horst und Hugo";
    s1.Remove(5, 8);
    AssertTrue("Remove(5, 8)", !::wcscmp(s1.PeekBuffer(), L"Horsto"));

    s1 = L"Horst und Hugo";
    s1.Remove(L"und ");
    AssertTrue("Remove \"und \"", !::wcscmp(s1.PeekBuffer(), L"Horst Hugo"));

    s1 = L"Horst";
    s1.Clear();
    AssertTrue("Cleared string is emtpy", s1.IsEmpty());

#ifdef _WIN32
    AssertTrue("Resource found.", s1.Load(101));
    AssertTrue("Load string from resource", !::wcscmp(s1.PeekBuffer(), L"Das ist ein Test."));
#endif /* _WIN32 */

    s1 = L"Horst Horst";
    s1.Replace(L"orst", L"ugo", 1);
    AssertTrue("Limited string replace", !::wcscmp(s1.PeekBuffer(), L"Hugo Horst"));

    s1 = L"Horst Horst";
    s1.Replace(L"orst", L"ugo", StringW::NO_LIMIT);
    AssertTrue("Unlimited string replace", !::wcscmp(s1.PeekBuffer(), L"Hugo Hugo"));

    s1 = L"Horst Horst";
    s1.Replace(L'o', L'i', 1);
    AssertTrue("Limited char replace", !::wcscmp(s1.PeekBuffer(), L"Hirst Horst"));

    s1 = L"Horst Horst";
    s1.Replace(L'o', L'i', StringW::NO_LIMIT);
    AssertTrue("Unlimited char replace", !::wcscmp(s1.PeekBuffer(), L"Hirst Hirst"));

    s1 = L"Hrst";
    s1.Insert(1, 'o');
    AssertTrue("Character insert at 1", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"orst";
    s1.Insert(0, L'H');
    AssertTrue("Character insert at 0", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Hors";
    s1.Insert(4, L't');
    AssertTrue("Character insert at end", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Ht";
    s1.Insert(1, L"ors");
    AssertTrue("String insert at 1", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"rst";
    s1.Insert(0, L"Ho");
    AssertTrue("String insert at 0", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Hor";
    s1.Insert(3, L"st");
    AssertTrue("String insert at end", !::wcscmp(s1.PeekBuffer(), L"Horst"));

    s1 = L"Hugo %d";
    s2.Format(&s1, 3);
    AssertTrue("Format with vislib format string", !::wcscmp(s2.PeekBuffer(), L"Hugo 3"));

	s1 = L"The quick brown fox jumps over the lazy dog.";
	s2 = s1;
	s2.Rot48();
	AssertNotEqual("Obfuscated is different", s1, s2);
	s2.Rot48();
	AssertEqual("Deobfuscated is equal to original", s1, s2);

    s1 = L"Tier";
    s2 = L"Tor";
    AssertEqual("Levenshtein distance.", s1.LevenshteinDistance(s2), StringW::Size(2));

    s1 = L"kitten";
    s2 = L"sitting";
    AssertEqual("Levenshtein distance.", s1.LevenshteinDistance(s2), StringW::Size(3));
}

void TestUTF8String(void) {
    vislib::StringA utf8;
    vislib::StringA strA;
    vislib::StringW strW;

    AssertEqual("CharTraitsW::ToAnsi", vislib::CharTraitsW::ToANSI(L'ä'), char('ä'));
    AssertEqual("CharTraitsA::ToUnicode", vislib::CharTraitsA::ToUnicode('Ö'), L'Ö');

    AssertEqual<unsigned int>("UTF8-Size of \"Horst\" is 6", vislib::UTF8Encoder::CalcUTF8Size("Horst"), 6);
    AssertEqual<unsigned int>("UTF8-Size of L\"Horst\" is 6", vislib::UTF8Encoder::CalcUTF8Size(L"Horst"), 6);
    AssertEqual<unsigned int>("UTF8-Size of \"Bäh\" is 5", vislib::UTF8Encoder::CalcUTF8Size("Bäh"), 5);
    AssertEqual<unsigned int>("UTF8-Size of L\"Bäh\" is 5", vislib::UTF8Encoder::CalcUTF8Size(L"Bäh"), 5);

    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, "This is a 7-Bit Ansi Test."));
    AssertEqual("UTF8 of \"This is a 7-Bit Ansi Test.\" correct.", utf8, "This is a 7-Bit Ansi Test.");

    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, L"This is a 7-Bit Ansi Test."));
    AssertEqual("UTF8 of L\"This is a 7-Bit Ansi Test.\" correct.", utf8, "This is a 7-Bit Ansi Test.");

    AssertEqual<int>("Size of UTF8 String is correct", vislib::UTF8Encoder::StringLength(utf8), int(strlen("This is a 7-Bit Ansi Test.")));

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strA, utf8));
    AssertEqual("decoding of UTF8 \"This is a 7-Bit Ansi Test.\" to ansi correct.", strA, "This is a 7-Bit Ansi Test.");

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strW, utf8));
    AssertEqual("decoding of UTF8 \"This is a 7-Bit Ansi Test.\" to unicode correct.", strW, L"This is a 7-Bit Ansi Test.");

    unsigned char tst0[] = {'B', 0xc3, 0xa4, 'h', '!', 0x00};
    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, "Bäh!"));
    AssertEqual("UTF8 of \"Bäh!\" correct.", utf8, reinterpret_cast<const char*>(tst0));

    AssertEqual<int>("Size of UTF8 String is correct", vislib::UTF8Encoder::StringLength(utf8), int(strlen("Bäh!")));

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strA, utf8));
    AssertEqual("decoding of UTF8 \"Bäh!\" to ansi correct.", strA, "Bäh!");

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strW, utf8));
    AssertEqual("decoding of UTF8 \"Bäh!\" to unicode correct.", strW, L"Bäh!");

    unsigned char tst1[] = {0xc3, 0xa4, 0xc3, 0xb6, 0xc3, 0xbc, 0xc3, 0x84, 0xc3, 0x96, 0xc3, 0x9c, 0xc3, 0x9f, 0x00};
    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, L"äöüÄÖÜß"));
    AssertEqual("UTF8 of L\"äöüÄÖÜß\" correct.", utf8, reinterpret_cast<const char*>(tst1));

    AssertEqual<int>("Size of UTF8 String is correct", vislib::UTF8Encoder::StringLength(utf8), int(strlen("äöüÄÖÜß")));

    //AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, "äöüÄÖÜß™"));
    //AssertEqual("UTF8 of \"äöüÄÖÜß™\" correct.", utf8, reinterpret_cast<const char*>(tst1));

    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, "äöüÄÖÜß"));
    tst1[14] = 0; // the ™-Character is destroied by the conversion from ansi to unicode with wsnprintf
    AssertEqual("UTF8 of \"äöüÄÖÜß\" correct.", utf8, reinterpret_cast<const char*>(tst1));

//    AssertTrue("Encode", vislib::UTF8Encoder::Encode(utf8, L"äöüÄÖÜß™"));

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strA, utf8));
    AssertEqual("decoding of UTF8 \"äöüÄÖÜß\" to ansi correct.", strA, "äöüÄÖÜß");

    AssertTrue("Decode", vislib::UTF8Encoder::Decode(strW, utf8));
    AssertEqual("decoding of UTF8 \"äöüÄÖÜß\" to unicode correct.", strW, L"äöüÄÖÜß");

}


/*
 * TestStringTokeniser
 */
void TestStringTokeniser(void) {
    printf("Testing StringTokeniser:\n");
    
    vislib::StringTokeniserA tokeniser("This is  a Test", " ");
    if (!AssertTrue("Tokeniser HasNext", tokeniser.HasNext())) return;
    AssertEqual("Token correct", tokeniser.Next(), "This");
    if (!AssertTrue("Tokeniser HasNext", tokeniser.HasNext())) return;
    AssertEqual("Token correct", tokeniser.Next(), "is");
    if (!AssertTrue("Tokeniser HasNext", tokeniser.HasNext())) return;
    AssertEqual("Token correct", tokeniser.Next(), "");
    if (!AssertTrue("Tokeniser HasNext", tokeniser.HasNext())) return;
    AssertEqual("Token correct", tokeniser.Next(), "a");
    if (!AssertTrue("Tokeniser HasNext", tokeniser.HasNext())) return;
    AssertEqual("Token correct", tokeniser.Next(), "Test");
    if (!AssertFalse("Tokeniser HasNext NOT", tokeniser.HasNext())) return;

    vislib::StringTokeniserA t1("ThisBLAHisBLAHaBLAHsecondBLAHTest", "BLAH");
    if (!AssertTrue("Tokeniser HasNext", t1.HasNext())) return;
    AssertEqual("Token correct", t1.Next(), "This");
    if (!AssertTrue("Tokeniser HasNext", t1.HasNext())) return;
    AssertEqual("Token correct", t1.Next(), "is");
    if (!AssertTrue("Tokeniser HasNext", t1.HasNext())) return;
    AssertEqual("Token correct", t1.Next(), "a");
    if (!AssertTrue("Tokeniser HasNext", t1.HasNext())) return;
    AssertEqual("Token correct", t1.Next(), "second");
    if (!AssertTrue("Tokeniser HasNext", t1.HasNext())) return;
    AssertEqual("Token correct", t1.Next(), "Test");
    if (!AssertFalse("Tokeniser HasNext NOT", t1.HasNext())) return;

    vislib::StringTokeniserA t2("Test with no match a all", "Hugo");
    if (!AssertTrue("Tokeniser HasNext", t2.HasNext())) return;
    AssertEqual("Token correct", t2.Next(), "Test with no match a all");
    if (!AssertFalse("Tokeniser HasNext NOT", t2.HasNext())) return;
}
