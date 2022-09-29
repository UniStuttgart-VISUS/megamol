/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>
#include <vector>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ParamSlot.h"

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
    inline void SetTexture(Texture tex) {
        this->tex_ = tex;
    }

    inline void SetTexture(void* id, float x, float y) {
        this->tex_.texture = id;
        this->tex_.x = x;
        this->tex_.y = y;
    }

    inline void SetFlipX(bool flip) {
        this->flip_x_ = flip;
    }

    inline void SetFlipY(bool flip) {
        this->flip_y_ = flip;
    }

    inline std::vector<core::AbstractSlot*> GetParameterSlots() {
        return {&this->show_inspector_, &this->select_texture_};
    }

    inline bool GetShowInspectorSlotValue() {
        return show_inspector_.Param<core::param::BoolParam>()->Value();
    }

    inline int GetSelectTextureSlotValue() {
        return select_texture_.Param<core::param::EnumParam>()->Value();
    }

    /*
     * ShowWindow
     * The main function responsible for creating the imgui window and the different scenes.
     */
    void ShowWindow();

    /**
     * Ctor
     */
    TextureInspector(const std::vector<std::string>& textures);

    /**
     * Ctor
     */
    TextureInspector();

    /**
     * Dtor
     */
    ~TextureInspector();

private:
    /*
     * Init
     * Initializes opengl and creates a context.
     */
    void Init();

    /** Slot to toggle the texture inspector window */
    core::param::ParamSlot show_inspector_;

    /** Slot to select a specific texture */
    core::param::ParamSlot select_texture_;

    Texture tex_;
    // ImGuiTexInspect::InspectorFlags flags_;
    unsigned int flags_;
    bool flip_x_;
    bool flip_y_;
    bool initiated_;
};

} // namespace megamol::mmstd_gl::special
