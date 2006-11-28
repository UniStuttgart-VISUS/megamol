/*
 * Console.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CONSOLE_H_INCLUDED
#define VISLIB_CONSOLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     */
    class Console {

    public:

        // TODO: documentation
        // TODO: Windows implementation.
        static int Run(const char *command, StringA *outStdOut = NULL, 
            StringA *outStdErr = NULL);

        /**
         * Write formatted text output to the standard output.
         *
         * @param fmt The string to write, possibly containing printf-style
         *            placeholders.
         * @param ... Values for the placeholders.
         */
        // TODO: Should Write throw a system exception in case of an error?
        // TODO: Should Write return a boolean to signal success?
        static void Write(const char *fmt, ...);

        /**
         * Write a line of formatted text output to the standard output. The 
         * linebreak will be added by the method.
         *
         * @param fmt The string to write, possibly containing printf-style
         *            placeholders.
         * @param ... Values for the placeholders.
         */
        static void WriteLine(const char *fmt, ...);

        /** Dtor. */
        ~Console(void);

    private:

        /** Disallow instances of this class. */
        Console(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_CONSOLE_H_INCLUDED */

