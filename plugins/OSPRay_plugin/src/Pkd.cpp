#include "stdafx.h"
#include "Pkd.h"
#include <stdint.h>
#include <iostream>
#include <thread>

#  define POS(idx,dim) pos(idx,dim)

using namespace megamol;

VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

VISLIB_FORCEINLINE unsigned char byteFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);

ospray::PkdBuilder::PkdBuilder() :
    megamol::stdplugin::datatools::AbstractParticleManipulator("outData", "inData"),
    inDataHash(0),
    outDataHash(0),
    numParticles(0),
    numInnerNodes(0) {
    model = std::make_shared<ParticleModel>();
}

ospray::PkdBuilder::~PkdBuilder() {
    Release();
}

bool ospray::PkdBuilder::manipulateData(
    core::moldyn::MultiParticleDataCall& outData,
    core::moldyn::MultiParticleDataCall& inData) {

    if ((inData.DataHash() != inDataHash) || (inData.FrameID() != frameID) || (inData.DataHash() == 0)) {
        inDataHash = inData.DataHash();
        outDataHash++;
        frameID = inData.FrameID();
    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);
    outData.SetFrameID(frameID);
    outData.SetDataHash(outDataHash);

    for (unsigned int i = 0; i <inData.GetParticleListCount(); i++) {
        auto &parts = inData.AccessParticles(i);
        auto &out = outData.AccessParticles(i);

        // empty the model
        this->model->position.clear();

        // put data the data into the model
        // and build the pkd tree
        this->model->fill(parts);
        this->build();


        out.SetVertexData(megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, &this->model->position[0].x, 12);
        out.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_NONE, NULL, 0);
    }

    return true;
}



void ospray::PkdBuilder::setDim(size_t ID, int dim) const {
#if DIM_FROM_DEPTH
    return;
#else
    ospcommon::vec3f &particle = (ospcommon::vec3f &)this->model->position[ID];
    int &pxAsInt = (int &)particle.x;
    pxAsInt = (pxAsInt & ~3) | dim;
#endif
}

inline size_t ospray::PkdBuilder::maxDim(const ospcommon::vec3f &v) const {
    const float maxVal = ospcommon::reduce_max(v);
    if (maxVal == v.x) {
        return 0;
    } else if (maxVal == v.y) {
        return 1;
    } else if (maxVal == v.z) {
        return 2;
    } else {
        assert(false && "Invalid max val index for vec!?");
        return -1;
    }
}


inline void ospray::PkdBuilder::swap(const size_t a, const size_t b) const {
    std::swap(model->position[a], model->position[b]);
    for (size_t i = 0; i<model->attribute.size(); i++)
        std::swap(model->attribute[i]->value[a], model->attribute[i]->value[b]);
    if (!model->type.empty())
        std::swap(model->type[a], model->type[b]);
}


void ospray::PkdBuilder::build() {
    //PING;
    assert(this->model != NULL);


    assert(!model->position.empty());
    numParticles = model->position.size();
    assert(numParticles <= (1ULL << 31));

#if 0
    cout << "#osp:pkd: TEST: RANDOMIZING PARTICLES" << endl;
    for (size_t i = numParticles - 1; i>0; --i) {
        size_t j = size_t(drand48()*i);
        if (i != j) swap(i, j);
    }
    cout << "#osp:pkd: RANDOMIZED" << endl;
#endif

    numInnerNodes = numInnerNodesOf(numParticles);

    // determine num levels
    numLevels = 0;
    size_t nodeID = 0;
    while (isValidNode(nodeID)) { ++numLevels; nodeID = leftChildOf(nodeID); }
    //PRINT(numLevels);

    const ospcommon::box3f &bounds = model->getBounds();
    std::cout << "#osp:pkd: bounds of model " << bounds << std::endl;
    std::cout << "#osp:pkd: number of input particles " << numParticles << std::endl;
    this->buildRec(0, bounds, 0);
}



void ospray::PkdBuilder::buildRec(const size_t nodeID,
    const ospcommon::box3f &bounds,
    const size_t depth) const {
    // if (depth < 4)
    // std::cout << "#osp:pkd: building subtree " << nodeID << std::endl;
    if (!hasLeftChild(nodeID))
        // has no children -> it's a valid kd-tree already :-)
        return;

    // we have at least one child.
    const size_t dim = this->maxDim(bounds.size());
    // if (depth < 4) { PRINT(bounds); printf("depth %ld-> dim %ld\n",depth,dim); }
    const size_t N = numParticles;

    if (!hasRightChild(nodeID)) {
        // no right child, but not a leaf emtpy. must have exactly one
        // child on the left. see if we have to swap, but otherwise
        // nothing to do.
        size_t lChild = leftChildOf(nodeID);
        if (POS(lChild, dim) > POS(nodeID, dim))
            swap(nodeID, lChild);
        // and done
        setDim(nodeID, dim);
        return;
    }

    {

        // we have a left and a right subtree, each of at least 1 node.
        SubtreeIterator l0(leftChildOf(nodeID));
        SubtreeIterator r0(rightChildOf(nodeID));

        SubtreeIterator l((size_t)l0); //(leftChildOf(nodeID));
        SubtreeIterator r((size_t)r0); //(rightChildOf(nodeID));

                                // size_t numSwaps = 0, numComps = 0;
        float rootPos = POS(nodeID, dim);
        while (1) {

            while (isValidNode(l, N) && (POS(l, dim) <= rootPos)) ++l;
            while (isValidNode(r, N) && (POS(r, dim) >= rootPos)) ++r;

            if (isValidNode(l, N)) {
                if (isValidNode(r, N)) {
                    // both mis-mathces valid, just swap them and go on
                    swap(l, r); ++l; ++r;
                    continue;
                } else {
                    // mis-match on left side, but nothing on right side to swap with: swap with root
                    // --> can't go on on right side any more, but can still compact matches on left
                    l0 = l;
                    ++l;
                    while (isValidNode(l, N)) {
                        if (POS(l, dim) <= rootPos) { swap(l0, l); ++l0; }
                        ++l;
                    }
                    swap(nodeID, l0); ++l0;
                    rootPos = POS(nodeID, dim);

                    l = l0;
                    r = r0;
                    continue;
                }
            } else {
                if (isValidNode(r, N)) {
                    // mis-match on left side, but nothing on right side to swap with: swap with root
                    // --> can't go on on right side any more, but can still compact matches on left
                    r0 = r;
                    ++r;
                    while (isValidNode(r, N)) {
                        if (POS(r, dim) >= rootPos) { swap(r0, r); ++r0; }
                        ++r;
                    }
                    swap(nodeID, r0); ++r0;
                    rootPos = POS(nodeID, dim);

                    l = l0;
                    r = r0;
                    continue;
                } else {
                    // no mis-match on either side ... done.
                    break;
                }
            }
        }
    }

    ospcommon::box3f lBounds = bounds;
    ospcommon::box3f rBounds = bounds;

    setDim(nodeID, dim);

    lBounds.upper[dim] = rBounds.lower[dim] = pos(nodeID, dim);

    if ((numLevels - depth) > 20) {
        std::thread lThread(&pkdBuildThread, new PKDBuildJob(this, leftChildOf(nodeID), lBounds, depth + 1));
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
        lThread.join();
    } else
    {
        buildRec(leftChildOf(nodeID), lBounds, depth + 1);
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
    }
}




uint32_t ospray::ParticleModel::getAtomTypeID(const std::string &name) {
    if (atomTypeByName.find(name) == atomTypeByName.end()) {
        if (name != "Default") std::cout << "New atom type '" + name + "'" << std::endl;
        ParticleModel::AtomType *a = new ParticleModel::AtomType(name);
        a->color = makeRandomColor(atomType.size());
        atomTypeByName[name] = atomType.size();
        atomType.push_back(a);
    }
    return atomTypeByName[name];
}

//! helper function for parser error recovery: 'clamp' all attributes to largest non-empty attribute
void ospray::ParticleModel::cullPartialData() {
    size_t largestCompleteSize = position.size();
    for (std::vector<Attribute *>::const_iterator it = attribute.begin();
        it != attribute.end(); it++)
        largestCompleteSize = std::min(largestCompleteSize, (*it)->value.size());

    if (position.size() > largestCompleteSize) {
        std::cout << "#osp:uintah: atoms w missing attribute(s): discarding" << std::endl;
        position.resize(largestCompleteSize);
    }
    if (type.size() > largestCompleteSize) {
        std::cout << "#osp:uintah: atoms w missing attribute(s): discarding" << std::endl;
        type.resize(largestCompleteSize);
    }
    for (std::vector<Attribute *>::const_iterator it = attribute.begin();
        it != attribute.end(); it++) {
        if ((*it)->value.size() > largestCompleteSize) {
            std::cout << "#osp:uintah: attribute(s) w/o atom(s): discarding" << std::endl;
            (*it)->value.resize(largestCompleteSize);
        }
    }
}

//! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
ospcommon::box3f ospray::ParticleModel::getBounds() const {
    ospcommon::box3f bounds = ospcommon::empty;
    for (size_t i = 0; i<position.size(); i++)
        bounds.extend(position[i]);
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

    // Color data type check
    if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
        colorLength = 4;
        //convertedColorType = OSP_FLOAT4;

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {
            ospcommon::vec3f pos;
            pos.x = ffaf(parts, loop * (vertexLength + colorLength) + 0);
            pos.y = ffaf(parts, loop * (vertexLength + colorLength) + 1);
            pos.z = ffaf(parts, loop * (vertexLength + colorLength) + 2);

            this->position.push_back(pos);
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
        // this colorType will be transformed to:
        //convertedColorType = OSP_FLOAT4;
        colorLength = 1;

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;
        std::vector<float> cd;

        for (size_t loop = 0; loop < parts.GetCount(); loop++) {
            ospcommon::vec3f pos;
            pos.x = ffaf(parts, loop * (vertexLength + colorLength) + 0);
            pos.y = ffaf(parts, loop * (vertexLength + colorLength) + 1);
            pos.z = ffaf(parts, loop * (vertexLength + colorLength) + 2);

            this->position.push_back(pos);
        }


    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
        colorLength = 3;

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;
        unsigned int iter = 0;
        for (size_t loop = 0; loop < parts.GetCount(); loop++) {
            ospcommon::vec3f pos;
            pos.x = ffaf(parts, loop * (vertexLength + colorLength) + 0);
            pos.y = ffaf(parts, loop * (vertexLength + colorLength) + 1);
            pos.z = ffaf(parts, loop * (vertexLength + colorLength) + 2);

            this->position.push_back(pos);
        }

    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
        vislib::sys::Log::DefaultLog.WriteError("File format not supported.");
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
        vislib::sys::Log::DefaultLog.WriteError("File format deprecated. Convert your data.");
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        colorLength = 0;

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;
        unsigned int iter = 0;
        for (size_t loop = 0; loop < parts.GetCount(); loop++) {
            ospcommon::vec3f pos;
            pos.x = ffaf(parts, loop * (vertexLength + colorLength) + 0);
            pos.y = ffaf(parts, loop * (vertexLength + colorLength) + 1);
            pos.z = ffaf(parts, loop * (vertexLength + colorLength) + 2);

            this->position.push_back(pos);
        }
    }
}

//! get attributeset of given name; create a new one if not yet exists */
ospray::ParticleModel::Attribute *ospray::ParticleModel::getAttribute(const std::string &name) {
    for (int i = 0; i<attribute.size(); i++)
        if (attribute[i]->name == name) return attribute[i];
    attribute.push_back(new Attribute(name));
    return attribute[attribute.size() - 1];
}

//! return if attribute of this name exists 
bool ospray::ParticleModel::hasAttribute(const std::string &name) {
    for (int i = 0; i<attribute.size(); i++)
        if (attribute[i]->name == name) return true;
    return false;
}

//! add one attribute value to set of attributes of given name
void ospray::ParticleModel::addAttribute(const std::string &name, float value) {
    ParticleModel::Attribute *a = getAttribute(name);
    a->value.push_back(value);
    a->minValue = std::min(a->minValue, value);
    a->maxValue = std::max(a->maxValue, value);
}

