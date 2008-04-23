/*
 * AbstractCursor2DEvent.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCURSOR2DEVENT_H_INCLUDED
#define VISLIB_ABSTRACTCURSOR2DEVENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractCursorEvent.h"


namespace vislib {
namespace graphics {

    /**
     * Abstract base class for cursor2d events
     */
    class AbstractCursor2DEvent: public AbstractCursorEvent {
    /*
     * This class is intentionally empty.
     */
    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCURSOR2DEVENT_H_INCLUDED */
