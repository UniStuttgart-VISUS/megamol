/*
 * TriMeshSTLWriter.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "TriMeshSTLWriter.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include <stdexcept>

namespace megamol {
namespace datatools_gl {
namespace io {
TriMeshSTLWriter::TriMeshSTLWriter() : AbstractSTLWriter(geocalls_gl::CallTriMeshDataGL::ClassName()) {}

TriMeshSTLWriter::~TriMeshSTLWriter() {}

bool TriMeshSTLWriter::create() {
    return true;
}

bool TriMeshSTLWriter::copy_info_upstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) {
    return true;
}

bool TriMeshSTLWriter::copy_info_downstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) {
    return true;
}

bool TriMeshSTLWriter::copy_data(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) {
    auto& incoming_call = dynamic_cast<geocalls_gl::CallTriMeshDataGL&>(caller);
    auto& outgoing_call = dynamic_cast<geocalls_gl::CallTriMeshDataGL&>(callee);

    incoming_call.SetObjects(outgoing_call.Count(), outgoing_call.Objects());

    return true;
}

bool TriMeshSTLWriter::write_data(core::AbstractGetData3DCall& callee) {
    auto& outgoing_call = dynamic_cast<geocalls_gl::CallTriMeshDataGL&>(callee);

    // Get data and save it to file
    try {
        // Sanity check
        if (outgoing_call.Objects()->GetVertexDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_NONE) {
            throw std::runtime_error("Illegal vertex data type");
        }
        if (outgoing_call.Objects()->GetNormalDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_NONE) {
            throw std::runtime_error("Illegal normal data type");
        }

        // Write file
        const std::size_t num_triangles = static_cast<std::size_t>(outgoing_call.Objects()->GetTriCount());

        if (outgoing_call.Objects()->GetVertexDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT) {
            if (outgoing_call.Objects()->GetNormalDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT) {
                if (outgoing_call.Objects()->GetTriDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerByte());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt16());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt32());
                } else {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerFloat());
                }
            } else {
                if (outgoing_call.Objects()->GetTriDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerByte());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt16());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt32());
                } else {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerFloat(),
                        outgoing_call.Objects()->GetNormalPointerDouble());
                }
            }
        } else {
            if (outgoing_call.Objects()->GetNormalDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT) {
                if (outgoing_call.Objects()->GetTriDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerByte());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt16());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerFloat(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt32());
                } else {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerFloat());
                }
            } else {
                if (outgoing_call.Objects()->GetTriDataType() == geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerByte());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt16());
                } else if (outgoing_call.Objects()->GetTriDataType() ==
                           geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32) {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerDouble(),
                        outgoing_call.Objects()->GetTriIndexPointerUInt32());
                } else {
                    AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
                        outgoing_call.Objects()->GetVertexPointerDouble(),
                        outgoing_call.Objects()->GetNormalPointerDouble());
                }
            }
        }
    } catch (const std::runtime_error& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Request for writing to STL file failed: %s", ex.what());

        return false;
    }

    return true;
}

void TriMeshSTLWriter::release() {}
} // namespace io
} // namespace datatools_gl
} // namespace megamol
