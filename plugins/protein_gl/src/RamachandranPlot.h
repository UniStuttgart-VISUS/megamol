/**
 * MegaMol
 * Copyright (c) 2017-2022, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/RamachandranDataCall.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glowl/glowl.h>

namespace megamol::protein_gl {
class RamachandranPlot : public mmstd_gl::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "RamachandranPlot";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Calculates the Ramachandran Plot of a given protein by plotting the dihedral angles of each amino acid";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    RamachandranPlot();

    /** Dtor. */
    ~RamachandranPlot() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The mouse button callback
     *
     * @param button The button the action was performed for
     * @param action The action that was performed with that button
     * @param mods Activated button modifiers
     *
     * @return True if the button press was processed, false otherwise
     */
    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    /**
     * The mouse move callback
     *
     * @param x The new x-coordinate of the mouse cursor
     * @param y The new y-coordinate of the mouse cursor
     *
     * @return True if the mouse move was processed, false otherwise
     */
    bool OnMouseMove(double x, double y) override;

private:
    /**
     * Computes the dihedral angles of the amino acids of a given protein
     *
     * @param mol Pointer the incoming molecular data call.
     */
    void computeDihedralAngles(protein_calls::MolecularDataCall& mol);

    /**
     * Computes the polygon positions of each available amino acid.
     */
    void computePolygonPositions(void);

    /**
     *  Tells whether a given point lies inside a given polygon
     *
     * @param polyVector A vector containing all points of the polygon ordererd clockwise or counterclockwise
     * @param inputPos The position to test against the polygon
     * @return True, if the point lies inside the polygon, false otherwise
     */
    bool locateInPolygon(const std::vector<glm::vec2>& polyVector, const glm::vec2 inputPos) const;

    /**
     * Computes the dihedral angle between four given vectors
     *
     * @param v1 The first vector
     * @param v2 The second vector
     * @param v3 The third vector
     * @param v4 The fourth vector
     * @return The dihedral angle between the four vectors in degrees. Range[-180, 180].
     */
    float dihedralAngle(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const glm::vec3& v4) const;

    /**
     *  Computes the angle between two given vectors. Normally, this should be doable by using the appropriate library.
     *  This angle computation however is somewhat special and follows the one used by biopython.
     *
     *  @param v1 The first vector
     *  @param v2 The second vector
     *
     *  @result The angle between v1 and v2 in radians
     */
    float angle(const glm::vec3& v1, const glm::vec3& v2) const;

    /** Input slot for the molecular data */
    core::CallerSlot moleculeInSlot_;

    core::param::ParamSlot showBoundingBoxParam_;
    core::param::ParamSlot pointSizeParam_;
    core::param::ParamSlot pointColorParam_;
    core::param::ParamSlot sureHelixColor_;
    core::param::ParamSlot unsureHelixColor_;
    core::param::ParamSlot sureSheetColor_;
    core::param::ParamSlot unsureSheetColor_;
    core::param::ParamSlot sureHelixPointColor_;
    core::param::ParamSlot unsureHelixPointColor_;
    core::param::ParamSlot sureSheetPointColor_;
    core::param::ParamSlot unsureSheetPointColor_;
    core::param::ParamSlot boundingBoxColor_;

    /** Vector containing both dihedral angles for all amino acids */
    std::vector<glm::vec2> angles_;
    /** Vector containing all probabilities */
    std::vector<float> probabilities_;
    /** The states of the rendered points */
    std::vector<protein_calls::RamachandranDataCall::PointState> pointStates_;
    /** The colors of the rendered points */
    std::vector<glm::vec3> pointColors_;

    /** Vector containing all bounding vertex shapes of definite helix locations */
    std::vector<std::vector<glm::vec2>> sureHelixPolygons_;
    /** Vector containing all bounding vertex shapes of definite sheet locations */
    std::vector<std::vector<glm::vec2>> sureSheetPolygons_;
    /** Vector containing all bounding vertex shapes of unsure helix locations */
    std::vector<std::vector<glm::vec2>> semiHelixPolygons_;
    /** Vector containing all bounding vertex shapes of unsure sheet locations */
    std::vector<std::vector<glm::vec2>> semiSheetPolygons_;

    /** The bounds of the graph */
    core::BoundingBoxes_2 bounds_;

    /** The passthrough shader to render all objects */
    std::unique_ptr<glowl::GLSLProgram> passthroughShader_;
    /** The stippling shader for the stippled lines */
    std::unique_ptr<glowl::GLSLProgram> stippleShader_;

    /** Buffer containing all positions to write */
    std::shared_ptr<glowl::BufferObject> positionBuffer_;
    /** Buffer containing all colors to write */
    std::shared_ptr<glowl::BufferObject> colorBuffer_;

    /** Vertex array handle */
    GLuint vao_;

    /** The font to draw the description */
    core::utility::SDFFont font_;
};
} // namespace megamol::protein_gl
