/*
 * WindowEscapeHotKeysUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED
#define MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED
#pragma once

#include "AbstractUILayer.h"
#include "gl/Window.h"

namespace megamol {
namespace console {
namespace gl {

    /** Utility class closing a window when ESC (or 'q') is pressed */
    class WindowEscapeHotKeysUILayer : public AbstractUILayer {
    public:
        WindowEscapeHotKeysUILayer(Window& wnd);
        virtual ~WindowEscapeHotKeysUILayer();
        virtual bool onKey(Key key, int scancode, KeyAction action, Modifiers mods);
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED */
