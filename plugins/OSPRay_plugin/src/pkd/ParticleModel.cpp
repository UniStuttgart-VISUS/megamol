#include "stdafx.h"
#include "pkd/ParticleModel.h"

#include "mmcore/utility/log/Log.h"

using namespace megamol;


VISLIB_FORCEINLINE float floatFromVoidArray(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    // const float* parts = static_cast<const float*>(p.GetVertexData());
    // return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}


VISLIB_FORCEINLINE unsigned char byteFromVoidArray(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetVertexData())[index];
}


VISLIB_FORCEINLINE float floatColFromVoidArray(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const float*>(p.GetColourData())[index];
}


VISLIB_FORCEINLINE unsigned char byteColFromVoidArray(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetColourData())[index];
}


typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);

typedef float(*floatColFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteColFromArrayFunc)(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


inline ospcommon::vec3f makeRandomColor(const int i) {
    const int mx = 13 * 17 * 43;
    const int my = 11 * 29;
    const int mz = 7 * 23 * 63;
    const uint32_t g = (i * (3 * 5 * 127) + 12312314);
    return ospcommon::vec3f((g % mx) * (1.f / (mx - 1)), (g % my) * (1.f / (my - 1)), (g % mz) * (1.f / (mz - 1)));
}


uint32_t ospray::ParticleModel::getAtomTypeID(const std::string& name) {
    if (atomTypeByName.find(name) == atomTypeByName.end()) {
        if (name != "Default") std::cout << "New atom type '" + name + "'" << std::endl;
        ParticleModel::AtomType* a = new ParticleModel::AtomType(name);
        a->color = makeRandomColor(atomType.size());
        atomTypeByName[name] = atomType.size();
        atomType.push_back(a);
    }
    return atomTypeByName[name];
}


//! helper function for parser error recovery: 'clamp' all attributes to largest non-empty attribute
void ospray::ParticleModel::cullPartialData() {
    size_t largestCompleteSize = position.size();
    for (std::vector<Attribute*>::const_iterator it = attribute.begin(); it != attribute.end(); it++)
        largestCompleteSize = std::min(largestCompleteSize, (*it)->value.size());

    if (position.size() > largestCompleteSize) {
        std::cout << "#osp:uintah: atoms w missing attribute(s): discarding" << std::endl;
        position.resize(largestCompleteSize);
    }
    if (type.size() > largestCompleteSize) {
        std::cout << "#osp:uintah: atoms w missing attribute(s): discarding" << std::endl;
        type.resize(largestCompleteSize);
    }
    for (std::vector<Attribute*>::const_iterator it = attribute.begin(); it != attribute.end(); it++) {
        if ((*it)->value.size() > largestCompleteSize) {
            std::cout << "#osp:uintah: attribute(s) w/o atom(s): discarding" << std::endl;
            (*it)->value.resize(largestCompleteSize);
        }
    }
}


//! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
ospcommon::box3f ospray::ParticleModel::getBounds() const {
    ospcommon::box3f bounds = ospcommon::empty;
    for (size_t i = 0; i < position.size(); i++) bounds.extend({position[i].x, position[i].y, position[i].z});
    return bounds;
}


void megamol::ospray::ParticleModel::fill(megamol::core::moldyn::SimpleSphericalParticles parts) {
    Attribute rgba("rgba");

    size_t vertexLength;
    size_t colorLength;

    // Vertex data type check
    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        vertexLength = 3;
    } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        vertexLength = 4;
    }

    auto const& parStore = parts.GetParticleStore();
    auto const& xAcc = parStore.GetXAcc();
    auto const& yAcc = parStore.GetYAcc();
    auto const& zAcc = parStore.GetZAcc();
    auto const& rAcc = parStore.GetCRAcc();
    auto const& gAcc = parStore.GetCGAcc();
    auto const& bAcc = parStore.GetCBAcc();
    auto const& aAcc = parStore.GetCAAcc();

    // Color data type check
    if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
        colorLength = 4;
        // convertedColorType = OSP_FLOAT4;

        /*floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;

        floatColFromArrayFunc fcfaf;
        fcfaf = floatColFromVoidArray;*/

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec3f pos;
            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);

            ospcommon::vec4f col;
            col.x = rAcc->Get_f(loop);
            col.y = gAcc->Get_f(loop);
            col.z = bAcc->Get_f(loop);
            col.w = aAcc->Get_f(loop);

            float const color = encodeColorToFloat(col);

            this->position.emplace_back(pos, color);
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
        // this colorType will be transformed to:
        // convertedColorType = OSP_FLOAT4;
        colorLength = 1;

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec3f pos;

            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);

            float const color = rAcc->Get_f(loop);

            this->position.emplace_back(pos, color);
        }


    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
        colorLength = 3;

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec3f pos;

            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);

            ospcommon::vec3f col;

            col.x = rAcc->Get_f(loop);
            col.y = gAcc->Get_f(loop);
            col.z = bAcc->Get_f(loop);

            float const color = encodeColorToFloat(col);

            this->position.emplace_back(pos, color);
        }

    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec3f pos;

            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);

            ospcommon::vec4uc col;

            col.x = rAcc->Get_u8(loop);
            col.y = gAcc->Get_u8(loop);
            col.z = bAcc->Get_u8(loop);
            col.w = aAcc->Get_u8(loop);

            float const color = encodeColorToFloat(col);

            this->position.emplace_back(pos, color);
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec3f pos;

            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);

            ospcommon::vec3uc col;

            col.x = rAcc->Get_u8(loop);
            col.y = gAcc->Get_u8(loop);
            col.z = bAcc->Get_u8(loop);

            float const color = encodeColorToFloat(col);

            this->position.emplace_back(pos, color);
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        colorLength = 0;

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {

            ospcommon::vec4f pos;

            pos.x = xAcc->Get_f(loop);
            pos.y = yAcc->Get_f(loop);
            pos.z = zAcc->Get_f(loop);
            pos.w = 0.0f;

            this->position.push_back(pos);
        }
    }
}


//! get attributeset of given name; create a new one if not yet exists */
ospray::ParticleModel::Attribute* ospray::ParticleModel::getAttribute(const std::string& name) {
    for (int i = 0; i < attribute.size(); i++)
        if (attribute[i]->name == name) return attribute[i];
    attribute.push_back(new Attribute(name));
    return attribute[attribute.size() - 1];
}


//! return if attribute of this name exists
bool ospray::ParticleModel::hasAttribute(const std::string& name) {
    for (int i = 0; i < attribute.size(); i++)
        if (attribute[i]->name == name) return true;
    return false;
}


//! add one attribute value to set of attributes of given name
void ospray::ParticleModel::addAttribute(const std::string& name, float value) {
    ParticleModel::Attribute* a = getAttribute(name);
    a->value.push_back(value);
    a->minValue = std::min(a->minValue, value);
    a->maxValue = std::max(a->maxValue, value);
}