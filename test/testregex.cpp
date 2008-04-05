/*
 * testregex.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testregex.h"
#include "testhelper.h"

#include "vislib/RegEx.h"


void TestRegEx(void) {
    using namespace vislib;
    RegExA reA;

    AssertException("Empty expressions are illegal", reA.Parse(""), RegExA::ParseException);

    AssertException("Illegal escape sequence", reA.Parse("\\="), RegExA::ParseException);

    AssertNoException("Escape sequence '\\\\'", reA.Parse("\\\\"));
    AssertNoException("Escape sequence '\\{'", reA.Parse("\\{"));
    AssertNoException("Escape sequence '\\}'", reA.Parse("\\}"));
    AssertNoException("Escape sequence '\\]'", reA.Parse("\\]"));
    AssertNoException("Escape sequence '\\['", reA.Parse("\\["));
    AssertNoException("Escape sequence '\\^'", reA.Parse("\\^"));
    AssertNoException("Escape sequence '\\,'", reA.Parse("\\,"));
    AssertNoException("Escape sequence '\\$'", reA.Parse("\\$"));
    AssertNoException("Escape sequence '\\.'", reA.Parse("\\."));
    AssertNoException("Escape sequence '\\-'", reA.Parse("\\-"));
    AssertNoException("Escape sequence '\\('", reA.Parse("\\("));
    AssertNoException("Escape sequence '\\)'", reA.Parse("\\)"));
    AssertNoException("Escape sequence '\\+'", reA.Parse("\\+"));
    AssertNoException("Escape sequence '\\?'", reA.Parse("\\?"));
    AssertNoException("Escape sequence '\\*'", reA.Parse("\\*"));
    AssertNoException("Escape sequence '\\|'", reA.Parse("\\|"));

    //AssertNoException("Abbreviation '\\w'", reA.Parse("\\w"));
    //AssertNoException("Abbreviation '\\W'", reA.Parse("\\W"));
    //AssertNoException("Abbreviation '\\s'", reA.Parse("\\s"));
    //AssertNoException("Abbreviation '\\S'", reA.Parse("\\S"));
    //AssertNoException("Abbreviation '\\d'", reA.Parse("\\d"));
    //AssertNoException("Abbreviation '\\D'", reA.Parse("\\D"));

}
