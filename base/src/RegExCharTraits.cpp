/*
 * RegExCharTraits.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/RegExCharTraits.h"

#include "vislib/UnsupportedOperationException.h"

////////////////////////////////////////////////////////////////////////////////
// Begin class RegExCharTraitsA

/*
 * vislib::RegExCharTraitsA::ABBREVIATIONS
 */
const vislib::RegExCharTraitsA::Char *vislib::RegExCharTraitsA::ABBREVIATIONS[]
    = { "w[a-zA-Z_0-9]",                    // Word character
        "W[^a-zA-Z_0-9]",                   // Non-word character
        "s[ \f\n\r\t\v]",                   // Whitespace character
        "S[^ \f\n\r\t\v]",                  // Non-whitespace character
        "d[0-9]",                           // Decimal digit
        "D[^0-9]",                          // No digit
        NULL
    };
    // These are the ATL abbreviations, but we want to have .NET abbreviations:
    //= { "a([a-zA-Z0-9])",                   // Alpha numeric
    //    "b([ \\t])",                        // Whitespace
    //    "c([a-zA-Z])",                      // Characters
    //    "d([0-9])",                         // Digits
    //    "h([0-9a-fA-F])",                   // Hex digit
    //    "n(\r|(\r?\n))",                    // Newline
    //    "q(\"[^\"]*\")|(\'[^\']*\')",       // Quoted string
    //    "w([a-zA-Z]+)",                     // Simple word
    //    "z([0-9]+)",                        // Integer
    //    NULL
    //};


/*
 * vislib::RegExCharTraitsA::ESCAPED_CHARS
 */
const vislib::RegExCharTraitsA::Char *vislib::RegExCharTraitsA::ESCAPED_CHARS
    = "\\}{][^,$.-)(+?*|";


/*
 * vislib::RegExCharTraitsA::RegExCharTraitsA
 */
vislib::RegExCharTraitsA::RegExCharTraitsA(void) : CharTraitsA() {
    throw UnsupportedOperationException("RegExCharTraitsA", __FILE__, __LINE__);
}

// End class RegExCharTraitsA
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin class RegExCharTraitsW

/*
 * vislib::RegExCharTraitsW::ABBREVIATIONS
 */
const vislib::RegExCharTraitsW::Char *vislib::RegExCharTraitsW::ABBREVIATIONS[]
    = { L"w[a-zA-Z_0-9]",                   // Word character
        L"W[^a-zA-Z_0-9]",                  // Non-word character
        L"s[ \f\n\r\t\v]",                  // Whitespace character
        L"S[^ \f\n\r\t\v]",                 // Non-whitespace character
        L"d[0-9]",                          // Decimal digit
        L"D[^0-9]",                         // No digit
        NULL
    };


/*
 * vislib::RegExCharTraitsW::ESCAPED_CHARS
 */
const vislib::RegExCharTraitsW::Char *vislib::RegExCharTraitsW::ESCAPED_CHARS
    = L"\\}{][^,$.-)(+?*|";


/*
 * vislib::RegExCharTraitsW::RegExCharTraitsW
 */
vislib::RegExCharTraitsW::RegExCharTraitsW(void) : CharTraitsW() {
    throw UnsupportedOperationException("RegExCharTraitsW", __FILE__, __LINE__);
}

// End class RegExCharTraitsW
////////////////////////////////////////////////////////////////////////////////
