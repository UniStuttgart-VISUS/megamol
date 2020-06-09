/*
 * ViewUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include <AbstractUILayer.h>

namespace megamol {
namespace console {

	using namespace megamol::input_events;

    /**
     * This UI layer propagates mouse events to the connected (core) view
     */
    class ViewUILayer : public AbstractUILayer {
    public:
        ViewUILayer(void * viewHandle);
        virtual ~ViewUILayer();

        virtual void OnResize(int w, int h);

		virtual bool OnKey(Key key, KeyAction action, Modifiers mods);
        virtual bool OnChar(unsigned int codePoint);
        virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods);
        virtual bool OnMouseMove(double x, double y);
        virtual bool OnMouseScroll(double x, double y);

    private:
        void *hView; // handle memory is owned by Window
    };

}
}
