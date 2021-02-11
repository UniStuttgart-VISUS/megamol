/*
 * AbstractCallRenderView.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Input.h"
#include "mmcore/thecam/camera.h"



namespace megamol {
namespace core {
namespace view {

/**
 * Call for rendering visual elements (from separate sources) into a single target, i.e.,
     * FBO-based compositing and cluster display.
 */
class MEGAMOLCORE_API AbstractCallRenderView : public AbstractCallRender {
public:

    /** Function index of 'render' */
    static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

    /** Function index of 'getExtents' */
    static const unsigned int CALL_EXTENTS = AbstractCallRender::FnGetExtents;

    /** Function index of 'freeze' */
    static const unsigned int CALL_FREEZE = 7;

    /** Function index of 'unfreeze' */
    static const unsigned int CALL_UNFREEZE = 8;

    /** Function index of 'ResetView' */
    static const unsigned int CALL_RESETVIEW = 9;

    /**
     * Answer the stereo projection eye
     *
     * @return the stereo projection eye
     */
    inline thecam::Eye GetEye(void) const {
        return this->eye;
    }

    /**
     * Answer the stereo projection type
     *
     * @return the stereo projection type
     */
    inline thecam::Projection_type GetProjectionType(void) const {
        return this->projType;
    }

    /**
     * Gets the input modifier
     *
     * @return The input modifier
     */
    inline Modifier InputModifier(void) const {
        return this->mod;
    }

    /**
     * Answers the flag indicating that the background information has been set
     *
     * @return 'true' if the background information has been set
     */
    inline bool IsBackgroundSet(void) const {
        return this->flagBkgnd;
    }

    /**
     * Answers the flag indicating that the projection information has been set
     *
     * @return 'true' if the projection information has been set
     */
    inline bool IsProjectionSet(void) const {
        return this->flagProj;
    }

    /**
     * Answers the flag indicating that the tile information has been set
     *
     * @return 'true' if the tile information has been set
     */
    inline bool IsTileSet(void) const {
        return this->flagTile;
    }

    /**
     * Answers the flag indicating that the viewport information has been set
     *
     * @return 'true' if the viewport information has been set
     */
    inline bool IsViewportSet(void) const {
        return true;
    }

    /**
     * Gets the button.
     *
     * @return The button
     */
    inline unsigned int MouseButton(void) const {
        return this->btn;
    }

    /**
     * Gets the 'down' flag.
     *
     * @return The 'down' flag
     */
    inline bool MouseButtonDown(void) const{
        return this->down;
    }

    /**
     * Gets the x coordinate.
     *
     * @return The x coordinate
     */
    inline float MouseX(void) const {
        return this->x;
    }

    /**
     * Gets the y coordinate.
     *
     * @return The y coordinate
     */
    inline float MouseY(void) const {
        return this->y;
    }

    /**
     * Propagates the parameters controlled by the frontend via 
     * mmcRenderViewContext to the call.
     *
     * @param context The context to get the data from.
     */
    inline void PropagateContext(const mmcRenderViewContext& context) {
        //this->SetGpuAffinity(context.GpuAffinity);
        this->SetInstanceTime(context.InstanceTime);
        this->SetTime(static_cast<float>(context.Time));
    }

    /**
     * Resets all flags
     */
    inline void ResetAll(void) {
        this->flagBkgnd = false;
        this->flagProj = false;
        this->flagTile = false;
    }

    /**
     * Resets the flag indicating that the background had been set.
     */
    inline void ResetBackground(void) {
        this->flagBkgnd = false;
    }

    /**
     * Resets the flag indicating that the projection had been set.
     */
    inline void ResetProjection(void) {
        this->flagProj = false;
    }

    /**
     * Resets the flag indicating that the tile had been set.
     */
    inline void ResetTile(void) {
        this->flagTile = false;
    }

    /**
     * Sets the projection information
     *
     * @param p The type of projection
     * @param e The eye used for stereo projections
     */
    inline void SetProjection(thecam::Projection_type p,
            thecam::Eye e = thecam::Eye::right) {
        this->flagProj = true;
        this->projType = p;
        this->eye = e;
    }

    /**
     * Sets the tile information
     *
     * @param fw The full width of the virtual viewport
     * @param fh The full height of the virtual viewport
     * @param x The x coordinate of the tile
     * @param y The y coordinate of the tile
     * @param w The width of the tile
     * @param h The height of the tile
     */
    inline void SetTile(float fw, float fh, float x, float y, float w, float h) {
        this->flagTile = true;
        this->width = fw;
        this->height = fh;
        this->tileX = x;
        this->tileY = y;
        this->tileW = w;
        this->tileH = h;
    }

    /**
     * Answer the height of the rendering tile
     *
     * @return The height of the rendering tile
     */
    inline float TileHeight(void) const {
        return this->tileH;
    }

    /**
     * Answer the width of the rendering tile
     *
     * @return The width of the rendering tile
     */
    inline float TileWidth(void) const {
        return this->tileW;
    }

    /**
     * Answer the x coordinate of the rendering tile
     *
     * @return The x coordinate of the rendering tile
     */
    inline float TileX(void) const {
        return this->tileX;
    }

    /**
     * Answer the y coordinate of the rendering tile
     *
     * @return The y coordinate of the rendering tile
     */
    inline float TileY(void) const {
        return this->tileY;
    }

    /**
     * Gets the height of the viewport in pixel.
     *
     * @return The height of the viewport in pixel
     */
    inline unsigned int ViewportHeight(void) const {
        return _framebuffer->height;
    }

    /**
     * Gets the width of the viewport in pixel.
     *
     * @return The width of the viewport in pixel
     */
    inline unsigned int ViewportWidth(void) const {
        return _framebuffer->width;
    }

    /**
     * Answer the height of the virtual viewport
     *
     * @return The height of the virtual viewport
     */
    inline float VirtualHeight(void) const {
        return this->height;
    }

    /**
     * Answer the width of the virtual viewport
     *
     * @return The width of the virtual viewport
     */
    inline float VirtualWidth(void) const {
        return this->width;
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to 'this'
     */
    AbstractCallRenderView& operator=(const AbstractCallRenderView& rhs);

protected:
    /**
     * Ctor.
     */
    AbstractCallRenderView(void);

private:

    /** The stereo projection eye */
    thecam::Eye eye;

    /** Flag indicating that the background colour information has been set */
    bool flagBkgnd : 1;

    /** Flag indicating that the projection information has been set */
    bool flagProj : 1;

    /** Flag indicating that the tile information has been set */
    bool flagTile : 1;

    /** The height of the virtual viewport */
    float height;

    /** The stereo projection type */
    thecam::Projection_type projType;

    /** The height of the rendering tile */
    float tileH;

    /** The width of the rendering tile */
    float tileW;

    /** The x coordinate of the rendering tile */
    float tileX;

    /** The y coordinate of the rendering tile */
    float tileY;

    /** The width of the virtual viewport */
    float width;

    /** The button */
    unsigned int btn;

    /**
     * Flag whether the button is pressed, or not, or the new input
     * modifier state
     */
    bool down;

    /** The x coordinate */
    float x;

    /** The y coordinate */
    float y;

    /** The input modifier to be set */
    Modifier mod;

    std::shared_ptr<CPUFramebuffer> _framebuffer;
};



} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
