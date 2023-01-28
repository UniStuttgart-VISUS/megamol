/*
 *Keyframe.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
#define MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
#pragma once


#include "mmcore/utility/JSONHelper.h"
#include "mmcore/view/Camera.h"
#include <glm/glm.hpp>


namespace megamol::cinematic {


/**
 * Keyframe Keeper.
 */
class Keyframe {
public:
    /** CTOR */
    Keyframe();

    Keyframe(float at, float st, core::view::Camera cam);

    /** DTOR */
    ~Keyframe();

    inline bool operator==(Keyframe const& rhs) {
        return ((this->camera_state == rhs.camera_state) && (this->anim_time == rhs.anim_time) &&
                (this->sim_time == rhs.sim_time));
    }

    inline bool operator!=(Keyframe const& rhs) {
        return (!((*this) == rhs));
    }

    // GET ----------------------------------------------------------------

    inline float GetAnimTime() const {
        return this->anim_time;
    }

    inline float GetSimTime() const {
        return this->sim_time; // (this->sim_time == 1.0f) ? (1.0f - 0.0000001f) : (this->sim_time);
    }

    inline core::view::Camera GetCamera() const {
        return this->camera_state;
    }

    // SET ----------------------------------------------------------------

    inline void SetAnimTime(float t) {
        this->anim_time = (t < 0.0f) ? (0.0f) : (t);
    }

    inline void SetSimTime(float t) {
        this->sim_time = glm::clamp(t, 0.0f, 1.0f);
    }

    inline void SetCameraState(const core::view::Camera& cam) {
        this->camera_state = cam;
    }

    // SERIALISATION ------------------------------------------------------

    bool Serialise(nlohmann::json& inout_json, size_t index);

    bool Deserialise(const nlohmann::json& in_json);

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    core::view::Camera camera_state;
    float sim_time; // Simulation time value is relative (always in [0,1])
    float anim_time;
};

} // namespace megamol::cinematic

#endif // MEGAMOL_CINEMATIC_KEYFRAME_H_INCLUDED
