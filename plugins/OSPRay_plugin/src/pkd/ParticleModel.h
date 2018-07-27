#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "ospcommon/box.h"


namespace megamol {
namespace ospray {

/*! complete input data for a particle model */
struct ParticleModel {


    ParticleModel() : radius(0) {}

    //! a set of attributes, one float per particle (with min/max info and logical name)
    struct Attribute {
        Attribute(const std::string& name)
            : name(name)
            , minValue(+std::numeric_limits<float>::infinity())
            , maxValue(-std::numeric_limits<float>::infinity()){};

        std::string name;
        float minValue, maxValue;
        std::vector<float> value;
    };
    struct AtomType {
        std::string name;
        ospcommon::vec3f color;

        AtomType(const std::string& name) : name(name), color(1, 0, 0) {}
    };

    //! list of all declared atom types
    std::vector<AtomType*> atomType;
    //! mapper that maps an atom type name to the ID in the 'atomType' vector
    std::map<std::string, int32_t> atomTypeByName;

    uint32_t getAtomTypeID(const std::string& name);

    std::vector<ospcommon::vec4f> position; //!< particle position + color encoded in 'w'
    std::vector<int> type;                  //!< 'type' of particle (e.g., the atom type for atomistic models)
    std::vector<Attribute*> attribute;

    void fill(megamol::core::moldyn::SimpleSphericalParticles parts);

    //! get attributeset of given name; create a new one if not yet exists */
    Attribute* getAttribute(const std::string& name);

    //! return if attribute of this name exists
    bool hasAttribute(const std::string& name);

    //! add one attribute value to set of attributes of given name
    void addAttribute(const std::string& attribName, float attribute);

    //! helper function for parser error recovery: 'clamp' all attributes to largest non-empty attribute
    void cullPartialData();

    //! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
    ospcommon::box3f getBounds() const;

    float encodeColorToFloat(ospcommon::vec4f const& col) {
        unsigned int r = static_cast<unsigned int>(col.x*255.f);
        unsigned int g = static_cast<unsigned int>(col.y*255.f);
        unsigned int b = static_cast<unsigned int>(col.z*255.f);
        unsigned int a = static_cast<unsigned int>(col.w*255.f);

        unsigned int color = (r << 24) + (g << 16) + (b << 8) + a;

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(ospcommon::vec4uc const& col) {
        unsigned int r = static_cast<unsigned int>(col.x);
        unsigned int g = static_cast<unsigned int>(col.y);
        unsigned int b = static_cast<unsigned int>(col.z);
        unsigned int a = static_cast<unsigned int>(col.w);

        unsigned int color = (r << 24) + (g << 16) + (b << 8) + a;

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(ospcommon::vec3f const& col) {
        unsigned int r = static_cast<unsigned int>(col.x*255.f);
        unsigned int g = static_cast<unsigned int>(col.y*255.f);
        unsigned int b = static_cast<unsigned int>(col.z*255.f);

        unsigned int color = (r << 24) + (g << 16) + (b << 8);

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float encodeColorToFloat(ospcommon::vec3uc const& col) {
        unsigned int r = static_cast<unsigned int>(col.x);
        unsigned int g = static_cast<unsigned int>(col.y);
        unsigned int b = static_cast<unsigned int>(col.z);

        unsigned int color = (r << 24) + (g << 16) + (b << 8);

        float ret = 0.0f;
        memcpy(&ret, &color, 4);

        return ret;
    }

    float radius; //!< radius to use (0 if not specified)
};

} /* end namespace ospray */
} /* end namespace megamol */
