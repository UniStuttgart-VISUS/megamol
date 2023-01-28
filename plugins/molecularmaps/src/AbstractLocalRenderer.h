/*
 * AbstractLocalRenderer.h
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_ABSTRACTLOCALRENDERER_H_INCLUDED
#define MMMOLMAPPLG_ABSTRACTLOCALRENDERER_H_INCLUDED
#pragma once

#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol {
namespace molecularmaps {

class AbstractLocalRenderer {
public:
    /** Ctor */
    AbstractLocalRenderer(void);

    /** Dtor */
    virtual ~AbstractLocalRenderer(void);

    /**
     * Initializes the renderer
     *
     * @return True on success, false otherwise
     */
    virtual bool create(void) = 0;

    /**
     * Invokes the rendering calls
     *
     * @param call The incoming rendering call containing the necessary camera information
     * @return True on success, false otherwise.
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call, bool lighting = true) = 0;

    /**
     * Frees all needed resources used by this renderer
     */
    virtual void release(void) = 0;

protected:
    /**
     * Private release function that invokes the release of the child classes
     */
    void Release(void);

    /** Hash of the last incoming data */
    SIZE_T lastDataHash;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_ABSTRACTLOCALRENDERER_H_INCLUDED */
