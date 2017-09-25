/*
 * ViewMouseUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include "AbstractUILayer.h"

namespace megamol {
namespace console {

    /**
     * This UI layer propagates mouse events to the connected (core) view
     */
    class ViewMouseUILayer : public AbstractUILayer {
    public:
        ViewMouseUILayer(gl::Window& wnd, void * viewHandle);
        virtual ~ViewMouseUILayer();

        virtual void onResize(int w, int h);
        virtual bool onMouseMove(double x, double y);
        virtual bool onMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods);
        virtual bool onMouseWheel(double x, double y);

    private:
        void *hView; // handle memory is owned by Window
    };

}
}
