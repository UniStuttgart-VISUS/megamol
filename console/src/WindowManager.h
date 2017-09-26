/*
 * WindowManager.h
 *
 * Copyright (C) 2008, 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_WINDOWMANAGER_H_INCLUDED
#define MEGAMOLCON_WINDOWMANAGER_H_INCLUDED
#pragma once

#include <memory>
#include <vector>

namespace megamol {
namespace console {

namespace gl {
    // forward declaration
    class Window;
}

    class WindowManager {
    public:

        /** The string to be prepended to window titles */
        static const char* const TitlePrefix;
        static const int TitlePrefixLength;

        /**
         * The singelton instance method.
         *
         * @return The only instance of this class.
         */
        static WindowManager& Instance(void);

        /** Dtor. */
        ~WindowManager(void);

        bool IsAlive(void) const;
        void Update(void);
        void Shutdown(void);

        bool InstantiatePendingView(void *hCore);

    private:

        /** Private ctor. */
        WindowManager(void);

        /** The viewing windows. */
        std::vector<std::shared_ptr<gl::Window> > windows;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_WINDOWMANAGER_H_INCLUDED */
