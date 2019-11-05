/*
 * AbstractInputScope.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "Input.h"

namespace megamol {
namespace render_api {

class AbstractInputScope {
public:
    /**
	 * This event handler can be reimplemented to receive key code events.
	 *
	 * @return true to stop propagation.
	 */
    virtual bool OnKey(Key key, KeyAction action, Modifiers mods) { return false; }
    
	/**
	 * This event handler can be reimplemented to receive unicode events.
	 *
	 * @return Returns true if the event was accepted (stopping propagation), otherwise false.
	 */
    virtual bool OnChar(unsigned int codePoint) { return false; }
    
	/**
	 * This event handler can be reimplemented to receive mouse button events.
	 *
	 * @return Returns true if the event was accepted (stopping propagation), otherwise false.
	 */
    virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) { return false; }
    
	/**
	 * This event handler can be reimplemented to receive mouse move events.
	 *
	 * @return Returns true if the event was accepted (stopping propagation), otherwise false.
	 */
    virtual bool OnMouseMove(double x, double y) { return false; }

	/**
	 * This event handler can be reimplemented to receive mouse scroll events.
	 *
	 * @return Returns true if the event was accepted (stopping propagation), otherwise false.
	 */
    virtual bool OnMouseScroll(double dx, double dy) { return false; }

protected:
    AbstractInputScope() = default;
    virtual ~AbstractInputScope() = default;
};

} /* end namespace render_api */
} /* end namespace megamol */

