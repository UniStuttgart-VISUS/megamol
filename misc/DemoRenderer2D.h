/*
 * DemoRenderer2D.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED
#define MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer2DModule.h"
#include "ModuleAutoDescription.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * A simple 2d renderer which just creates a circle
     *
     * TODO: Document!
     */
    class DemoRenderer2D : public view::Renderer2DModule {
    public:

        static const char *ClassName(void) {
            return "DemoRenderer2D";
        }

        static const char *Description(void) {
            return "Demo 2D-Renderer";
        }

        static bool IsAvailable(void) {
            return true;
        }

        DemoRenderer2D();

        virtual ~DemoRenderer2D();

    protected:

        virtual bool create(void);

        virtual bool GetExtents(view::CallRender2D& call);

        virtual bool Render(view::CallRender2D& call);

        virtual void release(void);

    private:

    };



} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED */
