/*
 * AbstractTileView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTTILEVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTTILEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Camera.h"
#include "mmcore_gl/view/AbstractOverrideViewGL.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core_gl {
namespace view {

/**
 * Abstract base class of override rendering views
 */
class AbstractTileViewGL : public AbstractOverrideViewGL {
public:
    /** Ctor. */
    AbstractTileViewGL(void);

    /** Dtor. */
    virtual ~AbstractTileViewGL(void);

    ///**
    // * Adjust the tile to match the window specified in 'context' if any.
    // *
    // * @param context The render context to adjust the tile from. It is safe
    // *                to pass NULL.
    // */
    //void AdjustTileFromContext(const mmcRenderViewContext *context);

protected:
    /** Initialises the tile view parameters */
    void initTileViewParameters(void);

    /** checks parameter slot changes */
    void checkParameters(void);

    /**
     * Answer the stereo projection type
     *
     * @return The stereo projection type
     */
    VISLIB_FORCEINLINE core::view::Camera::ProjectionType getProjType(void) const {
        return this->projType;
    }

    /**
     * Answer the height of the rendering tile
     *
     * @return The height of the rendering tile
     */
    VISLIB_FORCEINLINE float getTileH(void) const {
        return this->tileH;
    }

    /**
     * Answer the width of the rendering tile
     *
     * @return The width of the rendering tile
     */
    VISLIB_FORCEINLINE float getTileW(void) const {
        return this->tileW;
    }

    /**
     * Answer the x coordinate of the rendering tile
     *
     * @return The x coordinate of the rendering tile
     */
    VISLIB_FORCEINLINE float getTileX(void) const {
        return this->tileX;
    }

    /**
     * Answer the y coordinate of the rendering tile
     *
     * @return The y coordinate of the rendering tile
     */
    VISLIB_FORCEINLINE float getTileY(void) const {
        return this->tileY;
    }

    /**
     * Answer the height of the virtual viewport
     *
     * @return The height of the virtual viewport
     */
    VISLIB_FORCEINLINE float getVirtHeight(void) const {
        return this->virtHeight;
    }

    /**
     * Answer the width of the virtual viewport
     *
     * @return The width of the virtual viewport
     */
    VISLIB_FORCEINLINE float getVirtWidth(void) const {
        return this->virtWidth;
    }

    /**
     * Answer whether the conditions for overriding the tile when rendering
     * are met.
     *
     * @return true if the tile should be overridden, false otherwise.
     */
    inline bool hasTile(void) const {
        return ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0) && (this->getTileW() != 0) &&
                (this->getTileH() != 0));
    }

    /**
     * Set the tile configuration
     *
     * @param val The new tile value
     *
     * @return True on success
     */
    bool setTile(const vislib::TString& val);

private:
    /**
     * Packs the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void packMouseCoordinates(float& x, float& y);

    /** The stereo projection eye */
    core::param::ParamSlot eyeSlot;

    /** The stereo projection type */
    core::view::Camera::ProjectionType projType;

    /** The stereo projection type */
    core::param::ParamSlot projTypeSlot;

    /** The height of the rendering tile */
    float tileH;

    /** The rendering tile */
    core::param::ParamSlot tileSlot;

    /** The width of the rendering tile */
    float tileW;

    /** The x coordinate of the rendering tile */
    float tileX;

    /** The y coordinate of the rendering tile */
    float tileY;

    /** The height of the virtual viewport */
    float virtHeight;

    /** The virtual viewport size */
    core::param::ParamSlot virtSizeSlot;

    /** The width of the virtual viewport */
    float virtWidth;
};


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTTILEVIEW_H_INCLUDED */
