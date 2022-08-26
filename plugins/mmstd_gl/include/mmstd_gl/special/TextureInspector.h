/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

namespace megamol::mmstd_gl::special {

struct Texture {
    void* texture;
    float x; // size in x-direction
    float y; // size in y-direction
};

/**
 * Class implementing the texture inspector
 */
class TextureInspector {
public:

    void SetTexture(Texture tex) {
        this->tex_ = tex;
    }

    void SetTexture(void* id, float x, float y) {
        this->tex_.texture = id;
        this->tex_.x = x;
        this->tex_.y = y;
    }

    void SetFlipX(bool flip) {
        this->flip_x_ = flip;
    }

    void SetFlipY(bool flip) {
        this->flip_y_ = flip;
    }

    /*
    * ShowWindow
    * The main function responsible for creating the imgui window and the different scenes.
    */
    void ShowWindow();

    /**
     * Ctor
     */
    TextureInspector();

    /**
     * Dtor
     */
    ~TextureInspector();

private:
    /** The name of the view instance to be shot */
    //core::param::ParamSlot show_inspector_;

    /** Slot to select a specific texture */
    //core::param::ParamSlot which_texture_;

    /*
     * SceneColorFilters
     * An example showing controls to filter red, green and blue channels
     * independently. These controls are provided by the DrawColorChannelSelector
     * funtions.  The example also uses the DrawGridEditor function to allow the
     * user to control grid appearance.
     *
     * The Draw_* functions are provided for convenience, it is of course possible
     * to control these aspects programmatically.
     */
    void SceneColorFilters();

    /*
     * SceneColorMatrix
     * An example showing usage of the ColorMatrix.  See comments at the
     * declaration of CurrentInspector_SetColorMatrix for details.  This example
     * shows how to set the matrix directly, as well as how to use the
     * DrawColorMatrixEditor convenience function to draw ImGui controls to
     * manipulate it.
     */
    void SceneColorMatrix();

    void SceneTextureAnnotations();

    /*
     * SceneAlphaMode
     * Very simple example that calls DrawAlphaModeSelector to draw controls to
     * allow user to select alpha mode for the inpsector. See InspectorAlphaMode
     * enum for details on what the different modes are.
     */
    void SceneAlphaMode();

    /**
     * SceneWrapAndFilter
     * Scene showing the effect of the InspectorFlags_ShowWrap & InspectorFlags_NoForceFilterNearest flags.
     * See InspectorFlags_ enum for details on these flags.
     */
    void SceneWrapAndFilter();

    /*
    * Init
    * Initializes opengl and creates a context.
    */
    void Init();

    void (TextureInspector::*draw)();

    Texture tex_;
    // ImGuiTexInspect::InspectorFlags flags_;
    unsigned int flags_;
    bool flip_x_;
    bool flip_y_;
    bool initiated_;
};

} // namespace megamol::mmstd_gl
