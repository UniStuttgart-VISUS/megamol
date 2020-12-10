/*
 * TessellateBoundingBox.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef TESSELLATE_BOUNDING_BOX_H_INCLUDED
#define TESSELLATE_BOUNDING_BOX_H_INCLUDED

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

// plugin internal includes
#include "Utility.h"

namespace megamol {
namespace mesh {

    class TessellateBoundingBox : public core::Module {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) {
            return "TessellateBoundingBox";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "Tessellate a 3D Bounding Box into actual surface geometry";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        TessellateBoundingBox(void);

        /** Dtor. */
        virtual ~TessellateBoundingBox(void);

    protected:
        virtual bool create();
        virtual void release();


        core::CallerSlot _bounding_box_rhs_slot;
        core::CalleeSlot _mesh_lhs_slot;

        core::param::ParamSlot _subdiv_slot;
        core::param::ParamSlot _face_type_slot;

    private:
        bool InterfaceIsDirty();

        bool getMetaData(core::Call& call);
        bool getData(core::Call& call);

        // CallMesh stuff
        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
        mesh::MeshDataAccessCollection::IndexData _mesh_indices;
        mesh::MeshDataAccessCollection::PrimitiveType _mesh_type;
        uint32_t _version = 0;

        size_t _old_datahash;
        bool _recalc = false;

        std::array<uint32_t, 3> _subdivs;

        // store surface
        std::vector<std::array<float, 3>> _vertices;
        std::vector<std::array<float, 3>> _normals;
        std::vector<std::array<uint32_t, 4>> _faces;

    };

} // namespace probe
} // namespace megamol

#endif // !TESSELLATE_BOUNDING_BOX_H_INCLUDED
