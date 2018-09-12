/*
 * AbstractInputScope.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTINPUTSCOPE_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTINPUTSCOPE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Input.h"

namespace megamol {
namespace core {
namespace view {

class AbstractInputScope {
public:
    virtual bool OnKey(Key key, KeyAction action, Modifiers mods) { return false; }
    virtual bool OnChar(unsigned int codePoint) { return false; }
    virtual bool OnMouseMove(double x, double y) { return false; }
    virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) { return false; }
    virtual bool OnMouseScroll(double x, double y) { return false; }

protected:
    AbstractInputScope() = default;
    virtual ~AbstractInputScope() = default;
};

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTINPUTSCOPE_H_INCLUDED */
