/*
 * utility/HotFixFileName.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once
#include "AbstractUILayer.h"

namespace megamol {
namespace console {
namespace utility {

    /** Utility class closing a window when ESC (or 'q') is pressed */
    class HotFixFileName : public AbstractUILayer {
    public:
        HotFixFileName(gl::Window& wnd, void* hCore);
        virtual ~HotFixFileName();
        virtual bool onKey(Key key, int scancode, KeyAction action, Modifiers mods);
    private:
        void* hCore;
    };

}
}
}
