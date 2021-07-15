#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mmcore/param/ParamSlot.h"

#include "thermodyn/CallEvents.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

#include "CGAL/AABB_traits.h"
#include "CGAL/AABB_tree.h"
#include "CGAL/AABB_triangle_primitive.h"
#include "CGAL/Point_3.h"
#include "CGAL/Simple_cartesian.h"

namespace megamol::thermodyn {

class PrepareSurfaceEvents2 : public core::Module {
public:
    typedef CGAL::Simple_cartesian<double> K;
    typedef K::FT FT;
    typedef K::Point_3 Point;
    typedef K::Triangle_3 Triangle;
    typedef std::list<Triangle>::iterator Iterator;
    typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
    typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
    typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

    static const char* ClassName(void) {
        return "PrepareSurfaceEvents2";
    }

    static const char* Description(void) {
        return "PrepareSurfaceEvents2";
    }

    static bool IsAvailable(void) {
        return true;
    }

    PrepareSurfaceEvents2();

    virtual ~PrepareSurfaceEvents2();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return show_all_parts_slot_.IsDirty();
    }

    void reset_dirty() {
        show_all_parts_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& in_parts,
        stdplugin::datatools::table::TableDataCall& in_table, int frame_id);

    void prepare_maps(stdplugin::datatools::table::TableDataCall const& in_table);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot parts_in_slot_;

    core::CallerSlot table_in_slot_;

    core::param::ParamSlot show_all_parts_slot_;

    std::unordered_map<int /* start frame id */, std::vector<uint64_t> /* id */> frame_id_map_;

    std::unordered_map<uint64_t /* id */, std::tuple<int /* start frame id */, int /* end frame id */,
                                              glm::vec3 /* start pos */, glm::vec3 /* end pos */>>
        event_map_;

    std::shared_ptr<std::vector<glm::vec4>> events_;

    std::vector<std::vector<float>> part_data_;

    size_t parts_in_data_hash_ = std::numeric_limits<size_t>::max();

    size_t table_in_data_hash_ = std::numeric_limits<size_t>::max();

    int frame_id_ = -1;

    size_t data_out_hash_ = 0;

    size_t parts_out_hash_ = 0;
};

} // namespace megamol::thermodyn
