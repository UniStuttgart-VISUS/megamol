#include "PrecomputeGlyphTextures.h"

megamol::probe_gl::PrecomputeGlyphTextures::PrecomputeGlyphTextures(void)
        : _probes_rhs_slot("", "")
        , _textures_lhs_slot("", "")
        , _glyph_texture_data(std::make_shared<mesh::ImageDataAccessCollection>())
        , _glyph_texture_compute_shader(nullptr) {}

megamol::probe_gl::PrecomputeGlyphTextures::~PrecomputeGlyphTextures(void) {
    this->Release();
}

bool megamol::probe_gl::PrecomputeGlyphTextures::create() {

    //TODO create compute shader

    return true;
}

void megamol::probe_gl::PrecomputeGlyphTextures::release() {}

bool megamol::probe_gl::PrecomputeGlyphTextures::getMetaData(core::Call& call) {
    return true;
}

bool megamol::probe_gl::PrecomputeGlyphTextures::getData(core::Call& call) {

    //TODO check if probe data has update

    //TODO (re)allocate GPU textures for computation

    //TODO dispatch compute

    return true;
}
