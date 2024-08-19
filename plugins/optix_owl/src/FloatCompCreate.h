#pragma once

#include <exception>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <type_traits>

#include <owl/common/math/vec.h>

#include "floatcomp.h"

namespace megamol::optix_owl {
using namespace owl::common;

typedef union {
    float f;
    struct {
        unsigned int mantissa : 23;
        unsigned int exponent : 8;
        unsigned int sign : 1;
    } parts;
} float_cast;

struct QTPBufferBase {
    virtual ~QTPBufferBase() = default;
};

struct QTPBuffer_e5m15 : public QTPBufferBase {
    QTPBuffer_e5m15(size_t n) : buffer(n) {}
    std::vector<device::QTParticle_e5m15> buffer;
};

struct QTPBuffer_e5m15d : public QTPBufferBase {
    QTPBuffer_e5m15d(size_t n) : buffer(n) {}
    std::vector<device::QTParticle_e5m15d> buffer;
};

template<typename TYPE, bool BDEP = TYPE::dep, int BOFFSET = TYPE::offset>
inline void quantizeTree2(
    size_t P, vec3f const& basePos, device::Particle* particle, device::QTParticle* qparticle, size_t N) {
    if (P >= N)
        return;

    auto const& current = particle[P];

    auto diff = current.pos;
    diff.x = static_cast<double>(diff.x) - static_cast<double>(basePos.x);
    diff.y = static_cast<double>(diff.y) - static_cast<double>(basePos.y);
    diff.z = static_cast<double>(diff.z) - static_cast<double>(basePos.z);

    float_cast fc, fe;
    vec3f newBasePos;

    qparticle[P].dim = current.get_dim();
    fc.f = diff.x;
    qparticle[P].exp_x = (int) fc.parts.exponent - 127;
    qparticle[P].sign_x = fc.parts.sign;
    qparticle[P].m_x = fc.parts.mantissa >> BOFFSET;

    fe.parts.exponent = ((int) qparticle[P].exp_x) + 127u;
    fe.parts.sign = qparticle[P].sign_x;
    fe.parts.mantissa = ((unsigned int) qparticle[P].m_x) << BOFFSET;

    newBasePos.x = basePos.x + fe.f;

    fc.f = diff.y;
    qparticle[P].exp_y = (int) fc.parts.exponent - 127;
    qparticle[P].sign_y = fc.parts.sign;
    qparticle[P].m_y = fc.parts.mantissa >> BOFFSET;

    fe.parts.exponent = ((int) qparticle[P].exp_y) + 127u;
    fe.parts.sign = qparticle[P].sign_y;
    fe.parts.mantissa = ((unsigned int) qparticle[P].m_y) << BOFFSET;

    newBasePos.y = basePos.y + fe.f;

    fc.f = diff.z;
    qparticle[P].exp_z = (int) fc.parts.exponent - 127;
    qparticle[P].sign_z = fc.parts.sign;
    qparticle[P].m_z = fc.parts.mantissa >> BOFFSET;

    fe.parts.exponent = ((int) qparticle[P].exp_z) + 127u;
    fe.parts.sign = qparticle[P].sign_z;
    fe.parts.mantissa = ((unsigned int) qparticle[P].m_z) << BOFFSET;

    newBasePos.z = basePos.z + fe.f;

    if constexpr (BDEP) {
        quantizeTree2<TYPE>(lChild(P), newBasePos, particle, qparticle, N);
        quantizeTree2<TYPE>(rChild(P), newBasePos, particle, qparticle, N);
    } else {
        quantizeTree2<TYPE>(lChild(P), basePos, particle, qparticle, N);
        quantizeTree2<TYPE>(rChild(P), basePos, particle, qparticle, N);
    }
}

inline std::tuple<std::vector<unsigned int>, std::vector<unsigned int>, std::vector<unsigned int>> quantizeTree2(
    device::FloatCompType t, device::Particle* particle, device::PKDlet const& treelet, device::QTParticle* qparticle,
    device::FloatCompPKDlet& qtreelet) {
    if (treelet.end - treelet.begin == 0)
        //return std::vector<unsigned int>();
        return std::make_tuple<std::vector<unsigned int>, std::vector<unsigned int>, std::vector<unsigned int>>(
            {}, {}, {});

    qtreelet.bounds = treelet.bounds;
    qtreelet.begin = treelet.begin;
    qtreelet.end = treelet.end;

    switch (t) {
    /*case device::FloatCompType::E4M16: {
        qtreelet.basePos = treelet.bounds.lower;
        quantizeTree2<device::QTParticle_e4m16>(
            0, qtreelet.basePos, particle + treelet.begin, qparticle + treelet.begin, treelet.end - treelet.begin);
    } break;*/
    case device::FloatCompType::E5M15: {
        qtreelet.basePos = treelet.bounds.lower;
        quantizeTree2<device::QTParticle_e5m15>(
            0, qtreelet.basePos, particle + treelet.begin, qparticle + treelet.begin, treelet.end - treelet.begin);
    } break;
    /*case device::FloatCompType::E4M16D: {
        qtreelet.basePos = particle[treelet.begin].pos;
        quantizeTree2<device::QTParticle_e4m16d>(
            0, qtreelet.basePos, particle + treelet.begin, qparticle + treelet.begin, treelet.end - treelet.begin);
    } break;*/
    case device::FloatCompType::E5M15D: {
        qtreelet.basePos = particle[treelet.begin].pos;
        quantizeTree2<device::QTParticle_e5m15d>(
            0, qtreelet.basePos, particle + treelet.begin, qparticle + treelet.begin, treelet.end - treelet.begin);
    } break;
    default: /*{
        qtreelet.basePos = treelet.bounds.lower;
        quantizeTree2<device::QTParticle_e4m16>(
            0, qtreelet.basePos, particle + treelet.begin, qparticle + treelet.begin, treelet.end - treelet.begin);
    } break;*/
        throw std::runtime_error("unexpected FloatCompType");
    }

    std::vector<unsigned int> histo_x(256, 0);
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto idx = qparticle[i].exp_x + 128;
        histo_x[idx] += 1;
    }
    std::vector<unsigned int> histo_y(256, 0);
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto idx = qparticle[i].exp_y + 128;
        histo_y[idx] += 1;
    }
    std::vector<unsigned int> histo_z(256, 0);
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto idx = qparticle[i].exp_z + 128;
        histo_z[idx] += 1;
    }

    return std::make_tuple(histo_x, histo_y, histo_z);
}

inline void create_exp_maps(std::vector<device::QTParticle> const& input, std::unordered_set<char>& exponents_x,
    std::unordered_set<char>& exponents_y, std::unordered_set<char>& exponents_z, unsigned int num_idx) {
    for (auto const& qtp : input) {
        exponents_x.insert(qtp.exp_x);
        exponents_y.insert(qtp.exp_y);
        exponents_z.insert(qtp.exp_z);
    }
    if (exponents_x.size() > num_idx || exponents_y.size() > num_idx || exponents_z.size() > num_idx) {
        std::cout << "Exponents Overflow" << std::endl;
    }
}

inline bool create_exp_maps(device::FloatCompPKDlet& treelet, std::vector<device::QTParticle> const& input,
    char* out_exponents_x, char* out_exponents_y, char* out_exponents_z, unsigned int num_idx) {
    bool overflow = false;
    std::unordered_set<char> exponents_x;
    std::unordered_set<char> exponents_y;
    std::unordered_set<char> exponents_z;
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto const& qtp = input[i];
        exponents_x.insert(qtp.exp_x);
        exponents_y.insert(qtp.exp_y);
        exponents_z.insert(qtp.exp_z);
    }
    if (exponents_x.size() > num_idx || exponents_y.size() > num_idx || exponents_z.size() > num_idx) {
        std::cout << "Exponents Overflow" << std::endl;
        overflow = true;
    }
    std::copy(exponents_x.begin(), exponents_x.end(), out_exponents_x);
    std::copy(exponents_y.begin(), exponents_y.end(), out_exponents_y);
    std::copy(exponents_z.begin(), exponents_z.end(), out_exponents_z);

    return overflow;
}

template<typename QTP, int BEXP = QTP::exp, int BOFFSET = QTP::offset>
inline void convert_qlet(device::FloatCompPKDlet const& treelet, std::vector<device::QTParticle> const& input,
    std::vector<QTP>& output, char const* exponents_x, char const* exponents_y, char const* exponents_z) {
    auto num_idx = static_cast<unsigned int>(std::pow(2, BEXP));
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto const& in = input[i];
        auto& out = output[i];

        if (in.dim == 0) {
            out.dim_x = 1;
        }
        if (in.dim == 1) {
            out.dim_y = 1;
        }
        if (in.dim == 2) {
            out.dim_z = 1;
        }

        out.m_x = in.m_x;
        out.m_y = in.m_y;
        out.m_z = in.m_z;
        auto fit_x = std::find(exponents_x, exponents_x + num_idx, in.exp_x);
        out.exp_x = std::distance(exponents_x, fit_x);
        auto fit_y = std::find(exponents_y, exponents_y + num_idx, in.exp_y);
        out.exp_y = std::distance(exponents_y, fit_y);
        auto fit_z = std::find(exponents_z, exponents_z + num_idx, in.exp_z);
        out.exp_z = std::distance(exponents_z, fit_z);
    }
}

template<typename QTP, int BEXP = QTP::exp, int BOFFSET = QTP::offset, bool BSIGN = QTP::e_sign>
inline void convert_qlet_dep(device::FloatCompPKDlet const& treelet, std::vector<device::QTParticle> const& input,
    std::vector<QTP>& output, char const* exponents_x, char const* exponents_y, char const* exponents_z) {
    auto num_idx = static_cast<unsigned int>(std::pow(2, BEXP));
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        auto const& in = input[i];
        auto& out = output[i];

        if constexpr (BSIGN) {
            if (in.dim == 0) {
                out.dim_x = 1;
            } else if (in.dim == 1) {
                out.dim_y = 1;
            } else if (in.dim == 2) {
                out.dim_z = 1;
            }
            out.sign_x = in.sign_x;
            out.sign_y = in.sign_y;
            out.sign_z = in.sign_z;
        } else {
            if (in.dim == 0) {
                out.dim_a = 0;
                out.dim_b = 0;
            } else if (in.dim == 1) {
                out.dim_a = 1;
                out.dim_b = 0;
            } else if (in.dim == 2) {
                out.dim_a = 0;
                out.dim_b = 1;
            }

            auto const parent_id = parent(i - treelet.begin) + treelet.begin;
            auto const sep_dim = input[parent_id].dim;

            if (sep_dim == 0) {
                out.sign_a = in.sign_y;
                out.sign_b = in.sign_z;
            } else if (sep_dim == 1) {
                out.sign_a = in.sign_x;
                out.sign_b = in.sign_z;
            } else if (sep_dim == 2) {
                out.sign_a = in.sign_x;
                out.sign_b = in.sign_y;
            }
        }

        out.m_x = in.m_x;
        out.m_y = in.m_y;
        out.m_z = in.m_z;
        auto fit_x = std::find(exponents_x, exponents_x + num_idx, in.exp_x);
        out.exp_x = std::distance(exponents_x, fit_x);
        auto fit_y = std::find(exponents_y, exponents_y + num_idx, in.exp_y);
        out.exp_y = std::distance(exponents_y, fit_y);
        auto fit_z = std::find(exponents_z, exponents_z + num_idx, in.exp_z);
        out.exp_z = std::distance(exponents_z, fit_z);
    }
}

template<typename QTP, bool BDEP = QTP::dep, int BOFFSET = QTP::offset>
inline void sub_print(std::vector<vec3f>& diffs, std::vector<vec3f>& orgpos, std::vector<vec3f>& newpos,
    size_t P, vec3f refPos, device::Particle const* particles, QTP const* qparticles, size_t N,
    device::FloatCompPKDlet const& treelet, char const* exp_x_vec, char const* exp_y_vec, char const* exp_z_vec) {
    if (P >= N)
        return;

    auto const& base = particles[P];

    device::QTParticle current;
    if constexpr (/*std::is_same_v<QTP, device::QTParticle_e4m16> ||*/ std::is_same_v<QTP, device::QTParticle_e5m15>) {
        current = qparticles[P].getParticle();
    } else {
        current = qparticles[P].getParticle(P % 2 == 1, P == 0 ? 0 : particles[parent(P)].get_dim());
    }

    vec3f pos;

    int dim = current.dim;

    unsigned int x = 0;
    unsigned int sign_x = current.sign_x;
    x += (sign_x) << 31;
    char exp_x = exp_x_vec[current.exp_x];
    x += ((int) exp_x + 127u) << 23;
    x += (((unsigned int) current.m_x) << BOFFSET);

    float fx = *reinterpret_cast<float*>(&x);

    unsigned int y = 0;
    unsigned int sign_y = current.sign_y;
    y += (sign_y) << 31;
    char exp_y = exp_y_vec[current.exp_y];
    y += ((int) exp_y + 127u) << 23;
    y += (((unsigned int) current.m_y) << BOFFSET);

    float fy = *reinterpret_cast<float*>(&y);

    unsigned int z = 0;
    unsigned int sign_z = current.sign_z;
    z += (sign_z) << 31;
    char exp_z = exp_z_vec[current.exp_z];
    z += ((int) exp_z + 127u) << 23;
    z += (((unsigned int) current.m_z) << BOFFSET);

    float fz = *reinterpret_cast<float*>(&z);

    pos = vec3f(fx, fy, fz);
    pos += refPos;

    vec3d diff;
    diff.x = static_cast<double>(base.pos.x) - static_cast<double>(pos.x);
    diff.y = static_cast<double>(base.pos.y) - static_cast<double>(pos.y);
    diff.z = static_cast<double>(base.pos.z) - static_cast<double>(pos.z);

    diffs[P] = vec3f(diff);
    orgpos[P] = base.pos;
    newpos[P] = pos;

    if constexpr (BDEP) {
        sub_print<QTP>(
            diffs, orgpos, newpos, lChild(P), pos, particles, qparticles, N, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
        sub_print<QTP>(
            diffs, orgpos, newpos, rChild(P), pos, particles, qparticles, N, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
    } else {
        sub_print<QTP>(diffs, orgpos, newpos, lChild(P), refPos, particles, qparticles, N, treelet, exp_x_vec,
            exp_y_vec, exp_z_vec);
        sub_print<QTP>(diffs, orgpos, newpos, rChild(P), refPos, particles, qparticles, N, treelet, exp_x_vec,
            exp_y_vec, exp_z_vec);
    }
}

inline std::tuple<std::vector<vec3f>, std::vector<vec3f>, std::vector<vec3f>> unified_sub_print(
    device::FloatCompType selected_type, size_t P, vec3f refPos, device::Particle const* particles,
    std::shared_ptr<QTPBufferBase> const& qparticles, device::FloatCompPKDlet const& treelet, char const* exp_x_vec,
    char const* exp_y_vec, char const* exp_z_vec) {
    std::vector<vec3f> diffs(treelet.end - treelet.begin);
    std::vector<vec3f> orgpos(treelet.end - treelet.begin);
    std::vector<vec3f> newpos(treelet.end - treelet.begin);
    switch (selected_type) {
    case device::FloatCompType::E5M15: {
        auto buf = std::dynamic_pointer_cast<QTPBuffer_e5m15>(qparticles);
        sub_print<device::QTParticle_e5m15>(diffs, orgpos, newpos, 0, treelet.basePos, particles + treelet.begin,
            buf->buffer.data() + treelet.begin, treelet.end - treelet.begin, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
    } break;
    /*case device::FloatCompType::E4M16: {
        auto buf = std::dynamic_pointer_cast<QTPBuffer_e4m16>(qparticles);
        sub_print<device::QTParticle_e4m16>(diffs, orgpos, newpos, 0, treelet.basePos, particles + treelet.begin,
            buf->buffer.data() + treelet.begin, treelet.end - treelet.begin, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
    } break;*/
    case device::FloatCompType::E5M15D: {
        auto buf = std::dynamic_pointer_cast<QTPBuffer_e5m15d>(qparticles);
        sub_print<device::QTParticle_e5m15d>(diffs, orgpos, newpos, 0, treelet.basePos, particles + treelet.begin,
            buf->buffer.data() + treelet.begin, treelet.end - treelet.begin, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
    } break;
    /*case device::FloatCompType::E4M16D: {
        auto buf = std::dynamic_pointer_cast<QTPBuffer_e4m16d>(qparticles);
        sub_print<device::QTParticle_e4m16d>(diffs, orgpos, newpos, 0, treelet.basePos, particles + treelet.begin,
            buf->buffer.data() + treelet.begin, treelet.end - treelet.begin, treelet, exp_x_vec, exp_y_vec, exp_z_vec);
    } break;*/
    default:
        throw std::runtime_error("unexpected FloatCompType");
    }
    return std::make_tuple(diffs, orgpos, newpos);
}

} // namespace megamol::optix_owl
