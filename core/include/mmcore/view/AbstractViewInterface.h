/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AbstractInputScope.h"
#include "ImageWrapper.h"
#include "mmcore/Module.h"
#include "mmcore/view/Camera.h"

namespace megamol::core::view {

class AbstractViewInterface : public Module, public megamol::frontend_resources::AbstractInputScope {
public:
    enum class ViewDimension {
        NONE,
        VIEW_2D,
        VIEW_3D,
        OTHER,
    };

    explicit AbstractViewInterface(ViewDimension dim);

    virtual float DefaultTime(double instTime) const = 0;

    virtual void Resize(unsigned int width, unsigned int height) = 0;

    virtual void SetCamera(Camera camera, bool isMutable = true) = 0;

    virtual Camera GetCamera() const = 0;

    virtual std::string SampleCameraScenes(std::string camera_path_pattern, unsigned int num_samples) const = 0;

    ViewDimension const& GetViewDimension() const {
        return viewDimension_;
    }

    /**
     * Renders this AbstractView.
     * The View will use its own camera and framebuffer for the rendering exectuion
     *
     * @param time ...
     * @param instanceTime ...
     */
    using ImageWrapper = megamol::frontend_resources::ImageWrapper;
    virtual ImageWrapper Render(double time, double instanceTime) = 0;

    virtual ImageWrapper GetRenderingResult() const = 0;

protected:
    ViewDimension viewDimension_;
};

} // namespace megamol::core::view
