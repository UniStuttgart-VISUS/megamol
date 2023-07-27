/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "widgets/gl/ParameterGroupViewCubeWidget.h"

#include "graph/ParameterGroups.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::gui;


// *** Pickable Cube ******************************************************** //

megamol::gui::PickableCube::PickableCube() : image_up_arrow(), shader(nullptr) {}


bool megamol::gui::PickableCube::Draw(unsigned int picking_id, int& inout_selected_face_id,
    int& inout_selected_orientation_id, int& out_hovered_face_id, int& out_hovered_orientation_id,
    const glm::vec4& cube_orientation, core::utility::ManipVector_t& pending_manipulations) {

    assert(ImGui::GetCurrentContext() != nullptr);
    bool selected = false;

    // Create texture once
    if (!this->image_up_arrow.IsLoaded()) {
        this->image_up_arrow.LoadTextureFromFile(GUI_FILENAME_TEXTURE_VIEWCUBE_UP_ARROW);
    }

    // Create shader once -----------------------------------------------------
    if (this->shader == nullptr) {
        // INFO: IDs of the six cube faces are encoded via bit shift by face index of given parameter id.
        std::string vertex_src =
            "#version 130 \n"
            "\n"
            "uniform mat4 rot_mx; \n"
            "uniform mat4 model_mx; \n"
            "uniform mat4 proj_mx; \n"
            "uniform int selected_face_id; \n"
            "uniform int hovered_face_id; \n"
            "uniform int hovered_orientation_id; \n"
            "out vec2 tex_coord; \n"
            "flat out vec3 vertex_color; \n"
            "flat out vec3 original_color; \n"
            "flat out vec3 highlight_color; \n"
            "flat out int face_id; \n"
            "flat out int orientation_id; \n"
            "\n"
            "void main() { \n"
            "    // Vertex indices must fit enum order in megamol::core::view::View3D_2::DefaultView \n"
            "    const vec4 vertices[72] = vec4[72]( \n"
            "        // DEFAULTVIEW_FRONT = 0 \n"
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n"
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n"
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n"
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n"
            "        // DEFAULTVIEW_BACK = 1 \n"
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n"
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n"
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n"
            "        // DEFAULTVIEW_RIGHT = 2 \n"
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n"
            "        // DEFAULTVIEW_LEFT = 3 \n"
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n"
            "        // DEFAULTVIEW_TOP = 4 \n"
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        // DEFAULTVIEW_BOTTOM = 5 \n"
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n"
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n"
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0)); \n"
            "    \n"
            "    const vec3 colors[6] = vec3[6](vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), \n"
            "                                   vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0), \n"
            "                                   vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0)); \n"
            "    // IDs and indices \n"
            "    float vertex_index    = float(gl_VertexID); \n"
            "    float mod_index       = vertex_index - (12.0 * floor(vertex_index/12.0)); \n"
            "    int face_index        = int(gl_VertexID / 12); // in range [0-5] \n"
            "    face_id               = face_index; // same indices as in AbstractView3D::DefaultView \n"
            "    int orientation_index = int(floor(mod_index / 3.0)); // in range [0-3] \n"
            "    orientation_id        = orientation_index; // same indices as in AbstractView3D::DefaultOrientation \n"
            "    \n"
            "    // Vertex Color \n"
            "    original_color  = colors[face_index]; \n"
            "    highlight_color = clamp((original_color + vec3(0.5, 0.5, 0.5)), \n"
            "                             vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0)); \n"
            "    vertex_color    = original_color * 0.4; \n"
            "    if (selected_face_id == face_id) { \n"
            "        vertex_color = original_color * (0.5 + (0.5 - 0.5 * (float(orientation_index) / 3.0))); \n"
            "    } \n"
            "    if ((hovered_face_id == face_id) && (hovered_orientation_id == orientation_id)) { \n"
            "        vertex_color = highlight_color * (0.5 + (0.5 - 0.5 * (float(orientation_index) / 3.0))); \n"
            "    } \n"
            "    \n"
            "    // Up Arrow Texture \n"
            "    if ((mod_index == 0) || (mod_index == 4))      tex_coord = vec2(1.0, 0.0); \n"
            "    else if ((mod_index == 1) || (mod_index == 9)) tex_coord = vec2(0.0, 0.0); \n"
            "    else if ((mod_index == 3) || (mod_index == 7)) tex_coord = vec2(1.0, 1.0); \n"
            "    else if ((mod_index == 6) || (mod_index == 10)) tex_coord = vec2(0.0, 1.0); \n"
            "    else if ((mod_index == 2) || (mod_index == 5) || (mod_index == 8) || (mod_index == 11)) \n"
            "        tex_coord = vec2(0.5, 0.5); \n"
            "    \n"
            "    // Vertex Position \n"
            "    gl_Position = proj_mx * model_mx * rot_mx * vertices[gl_VertexID]; \n"
            "}";

        std::string fragment_src =
            "#version 130  \n"
            "#extension GL_ARB_explicit_attrib_location : require \n"
            "\n"
            "#define BIT_OFFSET_ID           7 \n"
            "#define BIT_OFFSET_FACE         2 \n"
            "#define BIT_OFFSET_ORIENTATION  0 \n"
            "\n"
            "#define FACE_FRONT                  0 \n"
            "#define FACE_BACK                   1 \n"
            "#define FACE_RIGHT                  2 \n"
            "#define FACE_LEFT                   3 \n"
            "#define FACE_TOP                    4 \n"
            "#define FACE_BOTTOM                 5 \n"
            "#define CORNER_TOP_LEFT_FRONT       6 \n"
            "#define CORNER_TOP_RIGHT_FRONT      7 \n"
            "#define CORNER_TOP_LEFT_BACK        8 \n"
            "#define CORNER_TOP_RIGHT_BACK       9 \n"
            "#define CORNER_BOTTOM_LEFT_FRONT    10 \n"
            "#define CORNER_BOTTOM_RIGHT_FRONT   11 \n"
            "#define CORNER_BOTTOM_LEFT_BACK     12 \n"
            "#define CORNER_BOTTOM_RIGHT_BACK    13 \n"
            "#define EDGE_TOP_FRONT              14 \n"
            "#define EDGE_TOP_LEFT               15 \n"
            "#define EDGE_TOP_RIGHT              16 \n"
            "#define EDGE_TOP_BACK               17 \n"
            "#define EDGE_BOTTOM_FRONT           18 \n"
            "#define EDGE_BOTTOM_LEFT            19 \n"
            "#define EDGE_BOTTOM_RIGHT           20 \n"
            "#define EDGE_BOTTOM_BACK            21 \n"
            "#define EDGE_FRONT_LEFT             22 \n"
            "#define EDGE_FRONT_RIGHT            23 \n"
            "#define EDGE_BACK_LEFT              24 \n"
            "#define EDGE_BACK_RIGHT             25 \n"
            "\n"
            "#define ORIENTATION_TOP             0 \n"
            "#define ORIENTATION_RIGHT           1 \n"
            "#define ORIENTATION_BOTTOM          2 \n"
            "#define ORIENTATION_LEFT            3 \n"
            "\n"
            "in vec2 tex_coord; \n"
            "flat in vec3 vertex_color; \n"
            "flat in vec3 original_color; \n"
            "flat in vec3 highlight_color; \n"
            "flat in int face_id; \n"
            "flat in int orientation_id; \n"
            "uniform sampler2D tex; \n"
            "uniform int picking_id; \n"
            "uniform int selected_face_id; \n"
            "uniform int hovered_face_id; \n"
            "layout(location = 0) out vec4 outFragColor; \n"
            "layout(location = 1) out vec2 outFragInfo; \n"
            "\n"
            "float supersample(const in vec2 uv, const in float w) { \n"
            "    return smoothstep(0.5 - w, 0.5 + w, texture(tex, uv).a); \n"
            "} \n"
            "\n"
            "float smoothalpha(const in float val, const in float max_val) { \n"
            "    float threshold = max_val * 0.75; \n"
            "    if (val < threshold) return 1.0; \n"
            "    return ((max_val - val) / threshold); \n"
            "} \n"
            "\n"
            "void process_edge_corner_hit(const in int id, out int new_id, \n"
            "                             in int selected_id, in int hovered_id, \n"
            "                             const in int CORNER_EDGE_FRONT_ID, const in int CORNER_EDGE_BACK_ID, \n"
            "                             const in int CORNER_EDGE_RIGHT_ID, const in int CORNER_EDGE_LEFT_ID, \n"
            "                             const in int CORNER_EDGE_TOP_ID, const in int CORNER_EDGE_BOTTOM_ID, \n"
            "                             const in float d_ec, const in float d_x, const in float d_y, \n"
            "                             out float selected_alpha, out float hovered_alpha) { \n"
            "    new_id         = -1; \n"
            "    selected_alpha = 0.0; \n"
            "    hovered_alpha  = 0.0; \n"
            "    \n"
            "    if (id == FACE_FRONT)       new_id = CORNER_EDGE_FRONT_ID; \n"
            "    else if (id == FACE_BACK)   new_id = CORNER_EDGE_BACK_ID; \n"
            "    else if (id == FACE_RIGHT)  new_id = CORNER_EDGE_RIGHT_ID; \n"
            "    else if (id == FACE_LEFT)   new_id = CORNER_EDGE_LEFT_ID; \n"
            "    else if (id == FACE_TOP)    new_id = CORNER_EDGE_TOP_ID; \n"
            "    else if (id == FACE_BOTTOM) new_id = CORNER_EDGE_BOTTOM_ID; \n"
            "    \n"
            "    if (((id == FACE_FRONT)   && (hovered_id == CORNER_EDGE_FRONT_ID))  || \n"
            "        ((id == FACE_BACK)    && (hovered_id == CORNER_EDGE_BACK_ID))   || \n"
            "        ((id == FACE_RIGHT)   && (hovered_id == CORNER_EDGE_RIGHT_ID))  || \n"
            "        ((id == FACE_LEFT)    && (hovered_id == CORNER_EDGE_LEFT_ID))   || \n"
            "        ((id == FACE_TOP)     && (hovered_id == CORNER_EDGE_TOP_ID))    || \n"
            "        ((id == FACE_BOTTOM)  && (hovered_id == CORNER_EDGE_BOTTOM_ID))) { \n"
            "        float dist = sqrt((d_x * d_x) + (d_y * d_y)); \n"
            "        hovered_alpha = smoothalpha(dist , d_ec); \n"
            "    } else if (((id == FACE_FRONT)   && (selected_id == CORNER_EDGE_FRONT_ID))  || \n"
            "        ((id == FACE_BACK)    && (selected_id == CORNER_EDGE_BACK_ID))   || \n"
            "        ((id == FACE_RIGHT)   && (selected_id == CORNER_EDGE_RIGHT_ID))  || \n"
            "        ((id == FACE_LEFT)    && (selected_id == CORNER_EDGE_LEFT_ID))   || \n"
            "        ((id == FACE_TOP)     && (selected_id == CORNER_EDGE_TOP_ID))    || \n"
            "        ((id == FACE_BOTTOM)  && (selected_id == CORNER_EDGE_BOTTOM_ID))) { \n"
            "        float dist = sqrt((d_x * d_x) + (d_y * d_y)); \n"
            "        selected_alpha = smoothalpha(dist , d_ec); \n"
            "    } \n"
            "} \n"
            "\n"
            "void main() { \n"
            "    vec4 out_color = vec4(vertex_color, 1.0); \n"
            "    \n"
            "    // Arrow Texture - supersample 4 extra points \n"
            "    float alpha = texture(tex, tex_coord).a; \n"
            "    if (alpha > 0.0) { \n"
            "        float smootingEdge = fwidth(alpha); \n"
            "        float dscale = 0.354; // half of 1/sqrt2; you can play with this \n"
            "        vec2 duv = dscale * (dFdx(tex_coord) + dFdy(tex_coord)); \n"
            "        vec4 box = vec4(tex_coord-duv, tex_coord+duv); \n"
            "        float asum = supersample(box.xy, smootingEdge) \n"
            "                   + supersample(box.zw, smootingEdge) \n"
            "                   + supersample(box.xw, smootingEdge) \n"
            "                   + supersample(box.zy, smootingEdge); \n"
            "        alpha = (alpha + 0.5 * asum) / 3.0; \n"
            "        vec3 tex_color = vertex_color * 0.5; \n"
            "        out_color = mix(vec4(vertex_color, 1.0), vec4(tex_color, 1.0), alpha); \n"
            "    } \n"
            "    \n"
            "    int final_face_id = face_id; \n"
            "    int final_orientation_id = orientation_id; \n"
            "    const float de = 0.1;  // must be in [0,1] \n"
            "    const float dc = 0.2;  // must be in [0,1] \n"
            "    float hovered_alpha  = 0.0; \n"
            "    float selected_alpha = 0.0; \n"
            "    // ----- Corners ----- \n"
            "    if ((tex_coord.x > (1.0 - dc)) && (tex_coord.y > (1.0 - dc))) {  // ----- BOTTOM RIGHT ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                CORNER_BOTTOM_RIGHT_FRONT, CORNER_BOTTOM_LEFT_BACK, \n"
            "                                CORNER_BOTTOM_RIGHT_BACK, CORNER_BOTTOM_LEFT_FRONT, \n"
            "                                CORNER_TOP_RIGHT_FRONT,  CORNER_BOTTOM_RIGHT_BACK, \n"
            "                                dc, (1.0 - tex_coord.x), (1.0 - tex_coord.y), \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "    } else if ((tex_coord.x < dc) && (tex_coord.y < dc)) {           // ----- TOP LEFT ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                CORNER_TOP_LEFT_FRONT, CORNER_TOP_RIGHT_BACK, CORNER_TOP_RIGHT_FRONT, \n"
            "                                CORNER_TOP_LEFT_BACK, CORNER_TOP_LEFT_BACK, CORNER_BOTTOM_LEFT_FRONT, \n"
            "                                dc, tex_coord.x, tex_coord.y, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "    } else if ((tex_coord.x < dc) && (tex_coord.y > (1.0 - dc))) {   // ----- BOTTOM LEFT ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                CORNER_BOTTOM_LEFT_FRONT, CORNER_BOTTOM_RIGHT_BACK, \n"
            "                                CORNER_BOTTOM_RIGHT_FRONT, CORNER_BOTTOM_LEFT_BACK, \n"
            "                                CORNER_TOP_LEFT_FRONT, CORNER_BOTTOM_LEFT_BACK, \n"
            "                                dc, tex_coord.x, (1.0 - tex_coord.y), \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "    } else if ((tex_coord.x > (1.0 - dc)) && (tex_coord.y < dc)) {   // ----- TOP RIGHT -----\n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                CORNER_TOP_RIGHT_FRONT, CORNER_TOP_LEFT_BACK, \n"
            "                                CORNER_TOP_RIGHT_BACK, CORNER_TOP_LEFT_FRONT, \n"
            "                                CORNER_TOP_RIGHT_BACK, CORNER_BOTTOM_RIGHT_FRONT, \n"
            "                                dc, (1.0 - tex_coord.x), tex_coord.y, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "    // ----- Edges ----- \n"
            "    } else if (tex_coord.x > (1.0 - de)) {                           // ----- RIGHT ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                EDGE_FRONT_RIGHT, EDGE_BACK_LEFT, EDGE_BACK_RIGHT, \n"
            "                                EDGE_FRONT_LEFT, EDGE_TOP_RIGHT, EDGE_BOTTOM_RIGHT, \n"
            "                                dc, (1.0 - tex_coord.x), 0.0, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "        if ((face_id == FACE_BOTTOM) && (final_face_id == EDGE_BOTTOM_RIGHT)) { \n"
            "            final_orientation_id = ORIENTATION_RIGHT; \n"
            "        } \n"
            "        if ((face_id == FACE_TOP) && (final_face_id == EDGE_TOP_RIGHT)) { \n"
            "            final_orientation_id = ORIENTATION_RIGHT; \n"
            "        } \n"
            "    } else if (tex_coord.x < de) {                                   // ----- LEFT ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                EDGE_FRONT_LEFT, EDGE_BACK_RIGHT, EDGE_FRONT_RIGHT, \n"
            "                                EDGE_BACK_LEFT, EDGE_TOP_LEFT, EDGE_BOTTOM_LEFT, \n"
            "                                dc, tex_coord.x, 0.0, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_TOP; \n"
            "        if ((face_id == FACE_BOTTOM) && (final_face_id == EDGE_BOTTOM_LEFT)) { \n"
            "            final_orientation_id = ORIENTATION_RIGHT; \n"
            "        } \n"
            "        if ((face_id == FACE_TOP) && (final_face_id == EDGE_TOP_LEFT)) { \n"
            "            final_orientation_id = ORIENTATION_RIGHT; \n"
            "        } \n"
            "    } else if (tex_coord.y > (1.0 - de)) {                           // ----- BOTTOM ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                EDGE_BOTTOM_FRONT, EDGE_BOTTOM_BACK, EDGE_BOTTOM_RIGHT, \n"
            "                                EDGE_BOTTOM_LEFT, EDGE_TOP_FRONT, EDGE_BOTTOM_BACK, \n"
            "                                dc, (1.0 - tex_coord.y), 0.0, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_RIGHT; \n"
            "    } else if (tex_coord.y < de) {                                   // ----- TOP ----- \n"
            "        process_edge_corner_hit(face_id, final_face_id, \n"
            "                                selected_face_id, hovered_face_id, \n"
            "                                EDGE_TOP_FRONT, EDGE_TOP_BACK, EDGE_TOP_RIGHT, \n"
            "                                EDGE_TOP_LEFT, EDGE_TOP_BACK, EDGE_BOTTOM_FRONT, \n"
            "                                dc, tex_coord.y, 0.0, \n"
            "                                selected_alpha, hovered_alpha); \n"
            "        final_orientation_id = ORIENTATION_RIGHT; \n"
            "        if ((face_id == FACE_TOP) && (final_face_id == EDGE_TOP_BACK)) { \n"
            "            final_orientation_id = ORIENTATION_LEFT; \n"
            "        } \n"
            "        if ((face_id == FACE_BACK) && (final_face_id == EDGE_TOP_BACK)) { \n"
            "            final_orientation_id = ORIENTATION_LEFT; \n"
            "        } \n"
            "    } \n"
            "    \n"
            "    // Color \n"
            "    if (hovered_alpha > 0.0) { \n"
            "        out_color = mix(out_color, vec4(highlight_color, 1.0), hovered_alpha); \n"
            "    } else if (selected_alpha > 0.0) { \n"
            "        out_color = mix(out_color, vec4(original_color, 1.0), selected_alpha); \n"
            "    } \n"
            "    outFragColor = out_color; \n"
            "    \n"
            "    // Ecoded ID \n"
            "    int encoded_id = int((picking_id           << BIT_OFFSET_ID)   | \n"
            "                         (final_face_id        << BIT_OFFSET_FACE) | \n"
            "                         (final_orientation_id << BIT_OFFSET_ORIENTATION)); \n"
            "    outFragInfo  = vec2(float(encoded_id), gl_FragCoord.z); \n"
            "} ";

        if (!core_gl::utility::RenderUtils::CreateShader(this->shader, vertex_src, fragment_src)) {
            return false;
        }
    }

    // Process pending manipulations ------------------------------------------
    const int BIT_OFFSET_ID = 7;
    const int BIT_OFFSET_FACE = 2;        // uses 5 bits
    const int BIT_OFFSET_ORIENTATION = 0; // uses 2 bits

    out_hovered_face_id = -1;
    out_hovered_orientation_id = -1;
    for (auto& manip : pending_manipulations) {
        // Check for right picking ID
        if (picking_id == (picking_id & (manip.obj_id >> BIT_OFFSET_ID))) {
            // Extract face and orientation ID
            int picked_face_id = static_cast<int>((manip.obj_id >> BIT_OFFSET_FACE) & 0b11111);
            int picked_orientation_id = static_cast<int>((manip.obj_id >> BIT_OFFSET_ORIENTATION) & 0b11);

            if (manip.type == core::utility::InteractionType::SELECT) {
                inout_selected_face_id = picked_face_id;
                inout_selected_orientation_id = picked_orientation_id;
                selected = true;
            } else if (manip.type == core::utility::InteractionType::HIGHLIGHT) {
                out_hovered_face_id = picked_face_id;
                out_hovered_orientation_id = picked_orientation_id;
            }
        }
    }

    // Draw -------------------------------------------------------------------
    GUI_GL_CHECK_ERROR

    // Create view/model and projection matrices
    const auto rotation = glm::inverse(
        glm::mat4_cast(glm::quat(cube_orientation.w, cube_orientation.x, cube_orientation.y, cube_orientation.z)));
    const float dist = 2.0f / std::tan(glm::radians(30.0f) / 2.0f);
    glm::mat4 model(1.0f);
    model[3][2] = -dist;
    const auto proj = glm::perspective(glm::radians(30.0f), 1.0f, 0.1f, 100.0f);

    // Set state
    const auto culling = glIsEnabled(GL_CULL_FACE);
    if (!culling) {
        glEnable(GL_CULL_FACE);
    }
    std::array<GLint, 4> viewport = {0, 0, 0, 0};
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    int size = (100 * static_cast<int>(megamol::gui::gui_scaling.Get()));
    int x = viewport[2] - size;
    int y = viewport[3] - size - static_cast<int>(ImGui::GetFrameHeightWithSpacing());
    glViewport(x, y, size, size);

    this->shader->use();

    auto texture_id = this->image_up_arrow.GetTextureID();
    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(this->shader->getUniformLocation("tex"), static_cast<GLint>(0));
    }

    this->shader->setUniform("rot_mx", rotation);
    this->shader->setUniform("model_mx", model);
    this->shader->setUniform("proj_mx", proj);
    this->shader->setUniform("selected_face_id", inout_selected_face_id);
    this->shader->setUniform("hovered_face_id", out_hovered_face_id);
    this->shader->setUniform("hovered_orientation_id", out_hovered_orientation_id);
    this->shader->setUniform("picking_id", static_cast<int>(picking_id));

    glDrawArrays(GL_TRIANGLES, 0, 72);

    if (texture_id != 0) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    glUseProgram(0);

    // Restore
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    if (!culling) {
        glDisable(GL_CULL_FACE);
    }

    GUI_GL_CHECK_ERROR

    return selected;
}


core::utility::InteractVector_t megamol::gui::PickableCube::GetInteractions(unsigned int id) const {

    core::utility::InteractVector_t interactions;
    interactions.emplace_back(
        core::utility::Interaction({core::utility::InteractionType::SELECT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    interactions.emplace_back(
        core::utility::Interaction({core::utility::InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    return interactions;
}


// *** Pickable Texture **************************************************** //

megamol::gui::PickableTexture::PickableTexture() : image_rotation_arrow(), shader(nullptr) {}


bool megamol::gui::PickableTexture::Draw(unsigned int picking_id, int selected_face_id, int& out_orientation_change,
    int& out_hovered_arrow_id, core::utility::ManipVector_t& pending_manipulations) {

    assert(ImGui::GetCurrentContext() != nullptr);
    bool selected = false;

    // Create texture once
    if (!this->image_rotation_arrow.IsLoaded()) {
        this->image_rotation_arrow.LoadTextureFromFile(GUI_FILENAME_TEXTURE_VIEWCUBE_ROTATION_ARROW);
    }

    // Create shader once -----------------------------------------------------
    if (this->shader == nullptr) {
        std::string vertex_src =
            "#version 450 \n"
            "uniform int picking_id; \n"
            "out vec2 tex_coord; \n"
            "flat out int encoded_id; \n"
            "void main() { \n"
            "    const vec4 vertices[12] = vec4[12]( \n"
            "        vec4(0.75, 0.75, -1.0, 1.0), vec4(1.0, 0.75, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), \n"
            "        vec4(0.75, 0.75, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(0.75, 1.0, -1.0, 1.0), \n"
            "        vec4(0.0, 0.75, -1.0, 1.0), vec4(0.25, 0.75, -1.0, 1.0), vec4(0.25, 1.0, -1.0, 1.0), \n"
            "        vec4(0.0, 0.75, -1.0, 1.0), vec4(0.25, 1.0, -1.0, 1.0), vec4(0.0, 1.0, -1.0, 1.0)); \n"
            "    const vec2 texcoords[12] = vec2[12]( \n"
            "        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), \n"
            "        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), \n"
            "        vec2(1.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 0.0), \n"
            "        vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 0.0)); \n"
            "    encoded_id  = int(picking_id << int(gl_VertexID / 6)); \n"
            "    gl_Position = vertices[gl_VertexID]; \n"
            "    tex_coord   = texcoords[gl_VertexID]; \n"
            "}";

        std::string fragment_src =
            "#version 450 \n"
            "#extension GL_ARB_explicit_attrib_location : require \n"
            "in vec2 tex_coord; \n"
            "flat in int encoded_id; \n"
            "uniform int selected_face_id; \n"
            "uniform sampler2D tex; \n"
            "uniform vec3 color; \n"
            "layout(location = 0) out vec4 outFragColor; \n"
            "layout(location = 1) out vec2 outFragInfo; \n"
            "float supersample(in vec2 uv, float w, float alpha) { \n"
            "    return smoothstep(0.5 - w, 0.5 + w, alpha); \n"
            "} \n"
            "void main() { \n"
            "    // Same colors as in vertex shader of pickable cube \n"
            "    const vec3 colors[6] = vec3[6](vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), \n"
            "                                   vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0), \n"
            "                                   vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0)); \n"
            "    vec4 tex_color = texture(tex, tex_coord); \n"
            "    float alpha = tex_color.a; \n"
            "    // Supersample - 4 extra points \n"
            "    float smootingEdge = fwidth(alpha); \n"
            "    float dscale = 0.354; // half of 1/sqrt2; you can play with this \n"
            "    vec2 duv = dscale * (dFdx(tex_coord) + dFdy(tex_coord)); \n"
            "    vec4 box = vec4(tex_coord-duv, tex_coord+duv); \n"
            "    float asum = supersample(box.xy, smootingEdge, alpha) \n"
            "               + supersample(box.zw, smootingEdge, alpha) \n"
            "               + supersample(box.xw, smootingEdge, alpha) \n"
            "               + supersample(box.zy, smootingEdge, alpha); \n"
            "    alpha = (alpha + 0.5 * asum) / 3.0; \n"
            "    if (alpha <= 0.0) discard; \n"
            "    if (selected_face_id < 6) outFragColor = vec4(colors[selected_face_id] * 0.75, alpha); \n"
            "    else outFragColor = vec4(0.75, 0.75, 0.75, alpha); \n"
            "    outFragInfo  = vec2(float(encoded_id), gl_FragCoord.z); \n"
            "} ";

        if (!core_gl::utility::RenderUtils::CreateShader(this->shader, vertex_src, fragment_src)) {
            return false;
        }
    }

    // Process pending manipulations ------------------------------------------
    out_hovered_arrow_id = 0;
    for (auto& manip : pending_manipulations) {
        int orientation_change = 0;
        if (picking_id == (manip.obj_id >> 0)) {
            orientation_change = -1;
        } else if (picking_id == (manip.obj_id >> 1)) {
            orientation_change = 1;
        }
        if (orientation_change != 0) {
            if (manip.type == core::utility::InteractionType::SELECT) {
                out_orientation_change = orientation_change;
                selected = true;
            } else if (manip.type == core::utility::InteractionType::HIGHLIGHT) {
                out_hovered_arrow_id = orientation_change;
            }
        }
    }

    // Draw -------------------------------------------------------------------
    GUI_GL_CHECK_ERROR

    // Set state
    const auto culling = glIsEnabled(GL_CULL_FACE);
    if (!culling) {
        glEnable(GL_CULL_FACE);
    }
    std::array<GLint, 4> viewport = {0, 0, 0, 0};
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    int size = (2 * 100 * static_cast<int>(megamol::gui::gui_scaling.Get()));
    int x = viewport[2] - size;
    int y = viewport[3] - size - static_cast<int>(ImGui::GetFrameHeightWithSpacing());
    glViewport(x, y, size, size);

    this->shader->use();

    auto texture_id = this->image_rotation_arrow.GetTextureID();
    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(static_cast<int>(this->shader->getUniformLocation("tex")), static_cast<GLint>(0));
    }

    this->shader->setUniform("selected_face_id", selected_face_id);
    this->shader->setUniform("picking_id", static_cast<int>(picking_id));

    // Arrow Color
    glm::vec3 color(0.6, 0.6, 0.6);
    this->shader->setUniform("color", color);

    glDrawArrays(GL_TRIANGLES, 0, 12);

    if (texture_id != 0) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    glUseProgram(0);

    // Restore
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    if (!culling) {
        glDisable(GL_CULL_FACE);
    }

    GUI_GL_CHECK_ERROR

    return selected;
}


core::utility::InteractVector_t megamol::gui::PickableTexture::GetInteractions(unsigned int id) const {

    core::utility::InteractVector_t interactions;
    interactions.emplace_back(
        core::utility::Interaction({core::utility::InteractionType::SELECT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    interactions.emplace_back(
        core::utility::Interaction({core::utility::InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    return interactions;
}


// *** Parameter Group View Cube Widget ************************************ //

megamol::gui::ParameterGroupViewCubeWidget::ParameterGroupViewCubeWidget()
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , tooltip()
        , cube_widget()
        , texture_widget()
        , last_presentation(param::AbstractParamPresentation::Presentation::Basic) {

    this->InitPresentation(ParamType_t::GROUP_3D_CUBE);
    this->name = "view";
}


bool megamol::gui::ParameterGroupViewCubeWidget::Check(bool only_check, ParamPtrVector_t& params) {

    bool param_cubeOrientation = false;
    bool param_defaultView = false;
    bool param_defaultOrientation = false;
    bool param_resetView = false;
    bool param_showCube = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "cubeOrientation") && (param_ptr->Type() == ParamType_t::VECTOR4F)) {
            param_cubeOrientation = true;
        } else if ((param_ptr->Name() == "defaultView") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultView = true;
        } else if ((param_ptr->Name() == "defaultOrientation") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultOrientation = true;
        } else if ((param_ptr->Name() == "resetView") && (param_ptr->Type() == ParamType_t::BUTTON)) {
            param_resetView = true;
        } else if ((param_ptr->Name() == "showViewCube") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_showCube = true;
        }
    }

    return (
        param_cubeOrientation && param_defaultView && param_showCube && param_defaultOrientation && param_resetView);
}


bool megamol::gui::ParameterGroupViewCubeWidget::Draw(ParamPtrVector_t params, const std::string& in_search,
    megamol::gui::Parameter::WidgetScope in_scope, core::utility::PickingBuffer* inout_picking_buffer,
    ImGuiID in_override_header_state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check required parameters ----------------------------------------------
    Parameter* param_cubeOrientation = nullptr;
    Parameter* param_defaultView = nullptr;
    Parameter* param_defaultOrientation = nullptr;
    Parameter* param_resetView = nullptr;
    Parameter* param_showCube = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "cubeOrientation") && (param_ptr->Type() == ParamType_t::VECTOR4F)) {
            param_cubeOrientation = param_ptr;
        } else if ((param_ptr->Name() == "defaultView") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultView = param_ptr;
        } else if ((param_ptr->Name() == "defaultOrientation") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultOrientation = param_ptr;
        } else if ((param_ptr->Name() == "resetView") && (param_ptr->Type() == ParamType_t::BUTTON)) {
            param_resetView = param_ptr;
        } else if ((param_ptr->Name() == "showViewCube") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_showCube = param_ptr;
        }
    }
    if ((param_cubeOrientation == nullptr) || (param_defaultView == nullptr) || (param_defaultOrientation == nullptr) ||
        (param_showCube == nullptr)) {
        utility::log::Log::DefaultLog.WriteError("[GUI] Unable to find all required parameters by name "
                                                 "for '%s' group widget. [%s, %s, line %d]\n",
            this->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = this->GetGUIPresentation();
    if (presentation != this->last_presentation) {
        param_showCube->SetValue((presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube));
        this->last_presentation = presentation;
    } else {
        if (std::get<bool>(param_showCube->GetValue())) {
            this->last_presentation = param::AbstractParamPresentation::Presentation::Group_3D_Cube;
            this->SetGUIPresentation(this->last_presentation);
        } else {
            this->last_presentation = param::AbstractParamPresentation::Presentation::Basic;
            this->SetGUIPresentation(this->last_presentation);
        }
    }

    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);

            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroups::DrawGroupedParameters(this->name, params, in_search, in_scope, nullptr, GUI_INVALID_ID);
            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {
            // GLOBAL

            if (inout_picking_buffer == nullptr) {
                utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to required picking buffer is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }

            ImGui::PushID(static_cast<int>(this->uid));

            auto default_view = std::get<int>(param_defaultView->GetValue());
            auto default_orientation = std::get<int>(param_defaultOrientation->GetValue());
            auto cube_orientation = std::get<glm::vec4>(param_cubeOrientation->GetValue());
            int hovered_face = -1;
            int hovered_orientation = -1;
            auto cube_picking_id = param_defaultView->UID(); // Using any parameter UID
            inout_picking_buffer->AddInteractionObject(
                cube_picking_id, this->cube_widget.GetInteractions(cube_picking_id));
            bool selected = this->cube_widget.Draw(cube_picking_id, default_view, default_orientation, hovered_face,
                hovered_orientation, cube_orientation, inout_picking_buffer->GetPendingManipulations());

            int hovered_arrow = 0;
            int orientation_change = 0;
            auto arrow_picking_id = param_defaultOrientation->UID(); // Using any other parameter UID
            inout_picking_buffer->AddInteractionObject(
                arrow_picking_id, this->texture_widget.GetInteractions(arrow_picking_id));
            this->texture_widget.Draw(arrow_picking_id, default_view, orientation_change, hovered_arrow,
                inout_picking_buffer->GetPendingManipulations());
            if (orientation_change < 0) {
                default_orientation = ((default_orientation - 1) < 0) ? (3) : (default_orientation - 1);
            } else if (orientation_change > 0) {
                default_orientation = ((default_orientation + 1) > 3) ? (0) : (default_orientation + 1);
            }

            if (selected) {
                param_resetView->ForceSetValueDirty();
            }
            param_defaultOrientation->SetValue(default_orientation);
            param_defaultView->SetValue(default_view);

            // Tooltip
            bool face_hovered = false;
            std::string tooltip_text;
            if (hovered_face >= 0) {
                switch (hovered_face) {
                case (DefaultView_t::DEFAULTVIEW_FACE_FRONT):
                    tooltip_text += "Face [Front]\nAxis [+Z]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_FACE_BACK):
                    tooltip_text += "Face [Back]\nAxis [-Z]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_FACE_RIGHT):
                    tooltip_text += "Face [Right]\nAxis [+X]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_FACE_LEFT):
                    tooltip_text += "Face [Left]\nAxis [-X]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_FACE_TOP):
                    tooltip_text += "Face [Top]\nAxis [+Y]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_FACE_BOTTOM):
                    tooltip_text += "Face [Bottom]\nAxis [-Y]";
                    face_hovered = true;
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_TOP_LEFT_FRONT):
                    tooltip_text += "Corner [Left][Top][Front]\nAxis [-X][+Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT):
                    tooltip_text += "Corner [Right][Top][Front]\nAxis [+X][+Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_TOP_LEFT_BACK):
                    tooltip_text += "Corner [Left][Top][Back]\nAxis [-X][+Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_TOP_RIGHT_BACK):
                    tooltip_text += "Corner [Right][Top][Back]\nAxis [+X][+Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT):
                    tooltip_text += "Corner [Left][Bottom][Front]\nAxis [-X][-Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT):
                    tooltip_text += "Corner [Right][Bottom][Front]\nAxis [+X][-Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK):
                    tooltip_text += "Corner [Left][Bottom][Back]\nAxis [-X][-Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK):
                    tooltip_text += "Corner [Right][Bottom][Back]\nAxis [+X][-Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_TOP_FRONT):
                    tooltip_text += "Edge [Top][Front]\nAxis [+Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_TOP_LEFT):
                    tooltip_text += "Edge [Top][Left]\nAxis [+Y][-X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_TOP_RIGHT):
                    tooltip_text += "Edge [Top][Right]\nAxis [+Y][+X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_TOP_BACK):
                    tooltip_text += "Edge [Top][Back]\nAxis [+Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BOTTOM_FRONT):
                    tooltip_text += "Edge [Bottom][Front]\nAxis [-Y][+Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BOTTOM_LEFT):
                    tooltip_text += "Edge [Bottom][Left]\nAxis [-Y][-X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BOTTOM_RIGHT):
                    tooltip_text += "Edge [Bottom][Right]\nAxis [-Y][+X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BOTTOM_BACK):
                    tooltip_text += "Edge [Bottom][Back]\nAxis [-Y][-Z]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_FRONT_LEFT):
                    tooltip_text += "Edge [Front][Left]\nAxis [+Z][-X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_FRONT_RIGHT):
                    tooltip_text += "Edge [Front][Right]\nAxis [+Z][+X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BACK_LEFT):
                    tooltip_text += "Edge [Back][Left]\nAxis [-Z][-X]";
                    break;
                case (DefaultView_t::DEFAULTVIEW_EDGE_BACK_RIGHT):
                    tooltip_text += "Edge [Back][Right]\nAxis [-Z][+X]";
                    break;
                default:
                    break;
                }
            }

            // Order is given by triangle order in shader of pickable cube
            if (face_hovered && (hovered_orientation >= 0)) {
                tooltip_text += " ";
                switch (hovered_orientation) {
                case (DefaultOrientation_t::DEFAULTORIENTATION_TOP):
                    tooltip_text += "\nRot [0 degree]";
                    break;
                case (DefaultOrientation_t::DEFAULTORIENTATION_RIGHT):
                    tooltip_text += "\nRot [+90 degree]";
                    break;
                case (DefaultOrientation_t::DEFAULTORIENTATION_BOTTOM):
                    tooltip_text += "\nRot [180 degree]";
                    break;
                case (DefaultOrientation_t::DEFAULTORIENTATION_LEFT):
                    tooltip_text += "\nRot [-90 degree]";
                    break;
                default:
                    break;
                }
            }

            /*
            if (hovered_arrow < 0) {
                tooltip_text = "Rotate Right";
            } else if (hovered_arrow > 0) {
                tooltip_text = "Rotate Left";
            }
            */

            if (!tooltip_text.empty()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip_text.c_str());
                ImGui::EndTooltip();
            }

            ImGui::PopID();

            return true;
        }
    }
    return false;
}
