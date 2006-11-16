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


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     */
    class Console {

    public:

        /**
         * Returns the object of the current console.
         *
         * @return A reference to the console object.
         */
        static Console& GetConsole(void);

        /** Dtor. */
        ~Console(void);

    protected:

    private:

    public: // delete me

        /** private Ctor. */
        Console(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_CONSOLE_H_INCLUDED */

