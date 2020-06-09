/*
 * WindowEscapeHotKeysUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED
#define MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED
#pragma once

#include "gl/Window.h"
#include <AbstractUILayer.h>

#include <functional>

namespace megamol {
namespace console {
namespace gl {

    /** Utility class closing a window when ESC is pressed */
    class WindowEscapeHotKeysUILayer : public AbstractUILayer {
    public:
        WindowEscapeHotKeysUILayer(Window& wnd);
        WindowEscapeHotKeysUILayer(std::function<void()> func);
        virtual ~WindowEscapeHotKeysUILayer();
        virtual bool OnKey(Key key, KeyAction action, Modifiers mods);

	private:
        Window* wndPtr = nullptr;
        std::function<void()> actionFunc;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_GL_WINDOWESCAPEHOTKEYSUILAYER_H_INCLUDED */
