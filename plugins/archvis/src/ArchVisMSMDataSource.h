/*
 * ArchVisMSMDataSource.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ARCH_VIS_MSM_DATASOURCE_H_INCLUDED
#define ARCH_VIS_MSM_DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

#include "vislib/net/Socket.h"

#include "mmcore/utility/SDFFont.h"

#include "mmcore/param/ParamSlot.h"

#include "ScaleModel.h"

#include <chrono>


namespace megamol {
namespace archvis {

class ArchVisMSMDataSource : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ArchVisMSMDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for visualizing SFB1244's 'Maﬂstabsmodell'"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    ArchVisMSMDataSource();
    ~ArchVisMSMDataSource();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    virtual bool getDataCallback(megamol::core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:
    typedef std::tuple<float, float, float> Node;
    typedef std::tuple<int, int, int, int> FloorElement;
    typedef std::tuple<int, int> BeamElement;
    typedef std::tuple<int, int> DiagonalElement;

    typedef vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> Mat4x4;
    typedef vislib::math::Vector<float, 3> Vec3;
    typedef vislib::math::Quaternion<float> Quat;

    struct TextLabelParticle {
        Vec3 position;
        std::string text;
        double age;
        Vec3 color;
    };

    struct PerObjectShaderParams {
        Mat4x4 transform;
        float force;
        Vec3 padding;
    };

    class DataPackage {
    public:
        DataPackage(int node_cnt, int input_element_cnt)
            : node_cnt(node_cnt)
            , input_element_cnt(input_element_cnt)
            , displacements_byteOffset(4)
            , forces_byteOffset(4 + node_cnt * 3 * 4)
            , byte_size(4 + node_cnt * 3 * 4 + input_element_cnt * 4)
            , raw_data(new uint8_t[byte_size]) {}

        ~DataPackage() { delete[] raw_data; }

        DataPackage(DataPackage const& other) = delete;
        DataPackage(DataPackage&& other) = delete;
        DataPackage& operator=(DataPackage const& rhs) = delete;
        DataPackage& operator=(DataPackage&& other) = delete;

        float getTime() { return reinterpret_cast<float*>(raw_data)[0]; }

        Vec3 getNodeDisplacement(int idx) {
            return Vec3(reinterpret_cast<float*>(raw_data + displacements_byteOffset + idx * 3 * 4)[0],
                reinterpret_cast<float*>(raw_data + displacements_byteOffset + idx * 3 * 4)[2],
                reinterpret_cast<float*>(raw_data + displacements_byteOffset + idx * 3 * 4)[1]);
        };

        float getElementForces(int idx) { return reinterpret_cast<float*>(raw_data + forces_byteOffset + idx * 4)[0]; }

        size_t getByteSize() { return byte_size; }

        uint8_t* data() { return raw_data; }

    private:
        int node_cnt;
        int input_element_cnt;
        size_t displacements_byteOffset;
        size_t forces_byteOffset;

        size_t byte_size;
        uint8_t* raw_data;
    };

    void parseNodeElementTable(std::string const& filename, std::vector<Node>& nodes,
        std::vector<FloorElement>& floor_elements, std::vector<BeamElement>& beam_elements,
        std::vector<DiagonalElement>& diagonal_elements);

    void parseNodeList(std::string const& filename, std::vector<Vec3>& node_positions);

    void parseElementList(std::string const& filename, std::vector<std::tuple<int, int, int, int, int>>& element_data);

    void parseInputElementList(std::string const& filename, std::vector<int>& input_elements);

    std::vector<std::string> parsePartsList(std::string const& filename);

    void updateMSMTransform();

    void spawnAndUpdateTextLabels();

    /** Representation of the scale model */
    ScaleModel m_scale_model;

    std::list<TextLabelParticle> m_text_particles;

    std::chrono::steady_clock::time_point m_last_spawn_time;
    std::chrono::steady_clock::time_point m_last_update_time;

    /** The mesh list file name */
    megamol::core::param::ParamSlot m_partsList_slot;

    /** The node list file name */
    megamol::core::param::ParamSlot m_nodes_slot;

    /** The element list file name */
    megamol::core::param::ParamSlot m_elements_slot;

    /** The node/element list file name */
    megamol::core::param::ParamSlot m_nodeElement_table_slot;

    /** The IP Adress for receiving sensor or simulation data */
    megamol::core::param::ParamSlot m_rcv_IPAddr_slot;

    /** The port for receiving sensor or simulation data */
    megamol::core::param::ParamSlot m_rcv_port_slot;

    /** The IP Adress for sending sensor or simulation data (to Unity) */
    megamol::core::param::ParamSlot m_snd_IPAddr_slot;

    /** The port for sending sensor or simulation data (to Unity) */
    megamol::core::param::ParamSlot m_snd_port_slot;

    /** The socket that receives the sensor or simulation data */
    vislib::net::Socket m_rcv_socket;
    bool m_rcv_socket_connected;

    /** The socket that sends the sensor or simulation data */
    vislib::net::Socket m_snd_socket;

    core::utility::SDFFont font;
};

} // namespace archvis
} // namespace megamol

#endif