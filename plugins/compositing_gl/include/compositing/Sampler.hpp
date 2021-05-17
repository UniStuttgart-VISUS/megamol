/*
 * Sampler.hpp
 *
 * MIT License
 * Copyright (c) 2021 Dominik Sellenthin
 */

#ifndef GLOWL_SAMPLER_HPP
#define GLOWL_SAMPLER_HPP

#include <string>
#include <array>
#include <vector>

#include "glowl/Exceptions.hpp"

#include "glowl/glinclude.h"

namespace glowl
{

    struct SamplerLayout
    {
    //SamplerLayout()
    //        : texture_min_filter(GL_NEAREST_MIPMAP_LINEAR)
    //        , texture_mag_filter(GL_LINEAR)
    //        , texture_min_lod(-1000)
    //        , texture_max_lod(1000)
    //        , texture_wrap_s(GL_REPEAT)
    //        , texture_wrap_t(GL_REPEAT)
    //        , texture_wrap_r(GL_REPEAT)
    //        , texture_border_color({0.f, 0.f, 0.f, 0.f})
    //        , texture_compare_mode(GL_NONE)
    //        , texture_compare_func(GL_NEVER) {}

    ///**
    // * \param texture_min_filter Specifies the minification filtering parameter (e.g. GL_NEAREST_MIPMAP_LINEAR)
    // * \param texture_mag_filter Specifies the magnification filtering parameter (GL_NEAREST or GL_LINEAR)
    // * \param texture_wrap_s Specifies the wrap parameter for texture coordinate s (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_wrap_t Specifies the wrap parameter for texture coordinate t (e.g. GL_CLAMP_TO_EDGE)
    // */
    //SamplerLayout(GLuint texture_min_filter, GLuint texture_mag_filter, GLuint texture_wrap_s, GLuint texture_wrap_t)
    //        : texture_min_filter(texture_min_filter)
    //        , texture_mag_filter(texture_mag_filter)
    //        , texture_wrap_s(texture_wrap_s)
    //        , texture_wrap_t(texture_wrap_t) {}

    ///**
    // * \param texture_min_filter Specifies the minification filtering parameter (e.g. GL_NEAREST_MIPMAP_LINEAR)
    // * \param texture_mag_filter Specifies the magnification filtering parameter (GL_NEAREST or GL_LINEAR)
    // * \param texture_wrap_s Specifies the wrap parameter for texture coordinate s (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_wrap_t Specifies the wrap parameter for texture coordinate t (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_wrap_r Specifies the wrap parameter for texture coordinate r (e.g. GL_CLAMP_TO_EDGE)
    // */
    //SamplerLayout(GLuint texture_min_filter, GLuint texture_mag_filter, GLuint texture_wrap_s, GLuint texture_wrap_t,
    //    GLuint texture_wrap_r)
    //        : texture_min_filter(texture_min_filter)
    //        , texture_mag_filter(texture_mag_filter)
    //        , texture_wrap_s(texture_wrap_s)
    //        , texture_wrap_t(texture_wrap_t)
    //        , texture_wrap_r(texture_wrap_r) {}

    ///**
    // * \param texture_min_filter Specifies the minification filtering parameter (e.g. GL_NEAREST_MIPMAP_LINEAR)
    // * \param texture_mag_filter Specifies the magnification filtering parameter (GL_NEAREST or GL_LINEAR)
    // * \param texture_min_lod Specifies the magnification filtering parameter (GL_NEAREST or GL_LINEAR)
    // * \param texture_max_lod Specifies the magnification filtering parameter (GL_NEAREST or GL_LINEAR)
    // * \param texture_wrap_s Specifies the wrap parameter for texture coordinate s (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_wrap_t Specifies the wrap parameter for texture coordinate t (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_wrap_r Specifies the wrap parameter for texture coordinate r (e.g. GL_CLAMP_TO_EDGE)
    // * \param texture_border_color Specifies four values that define the border values that should be used for border
    // * texels \param texture_compare_mode Specifies the texture comparison mode for currently bound depth texture
    // * (GL_COMPARE_REF_TO_TEXTURE or GL_NONE) \param texture_compare_func Specifies the comparison operator when
    // * texture_compare_mode is used (e.g. GL_LEQUAL)
    // */
    //SamplerLayout(GLuint texture_min_filter, GLuint texture_mag_filter, GLint texture_min_lod, GLint texture_max_lod,
    //    GLuint texture_wrap_s, GLuint texture_wrap_t, GLuint texture_wrap_r, std::array<GLfloat, 4> col,
    //    GLuint comparemode, GLuint comparefunc)
    //        : texture_min_filter(texture_min_filter)
    //        , texture_mag_filter(texture_mag_filter)
    //        , texture_min_lod(texture_min_lod)
    //        , texture_max_lod(texture_max_lod)
    //        , texture_wrap_s(texture_wrap_s)
    //        , texture_wrap_t(texture_wrap_t)
    //        , texture_wrap_r(texture_wrap_r)
    //        , texture_border_color(col)
    //        , texture_compare_mode(comparemode)
    //        , texture_compare_func(comparefunc) {}


    //GLuint texture_min_filter;
    //GLuint texture_mag_filter;
    //GLint texture_min_lod;
    //GLint texture_max_lod;
    //GLuint texture_wrap_s;
    //GLuint texture_wrap_t;
    //GLuint texture_wrap_r;
    //std::array<GLfloat, 4> texture_border_color;
    //GLuint texture_compare_mode;
    //GLuint texture_compare_func;

     /**
     * \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {...}, ...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLint>> const& int_params) : int_parameters(int_params) {}

    /**
     * \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {...}, ...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLint>>&& int_params) : int_parameters(int_params) {}

    /**
     * \param float_parameters A list of float texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_BORDER_COLOR, {1.f, 0.f, 0.f, 1.f}}, {...},...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLfloat>> const& float_params) : float_parameters(float_params) {}

    /**
     * \param float_parameters A list of float texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_BORDER_COLOR, {1.f, 0.f, 0.f, 1.f}}, {...},...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLfloat>>&& float_params) : float_parameters(float_params) {}

    /**
     * \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {...}, ...}
     * \param float_parameters A list of float texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_BORDER_COLOR, {1.f, 0.f, 0.f, 1.f}}, {...},...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLint>> const& int_params,
        std::vector<std::pair<GLenum, GLfloat>> const& float_params)
            : int_parameters(int_params)
            , float_parameters(float_params) {}

    /**
     * \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {...}, ...}
     * \param float_parameters A list of float texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_BORDER_COLOR, {1.f, 0.f, 0.f, 1.f}}, {...},...}
     */
    SamplerLayout(std::vector<std::pair<GLenum, GLint>>&& int_params,
        std::vector<std::pair<GLenum, GLfloat>>&& float_params)
            : int_parameters(int_params)
            , float_parameters(float_params) {}

     std::array<GLfloat, 4> border_color;   
     std::vector<std::pair<GLenum, GLint>> int_parameters;
     std::vector<std::pair<GLenum, GLfloat>> float_parameters;
    };

    /**
     * \class Sampler
     *
     * \brief Class for OpenGL sampler Objects
     *
     * \author Dominik Sellenthin
     */
    class Sampler {
    public:
        Sampler(std::string id) : m_id(id) {}

        Sampler(std::string id, SamplerLayout const& layout) : m_id(id) {
            glCreateSamplers(1, &m_name);

            for (const auto& p : layout.int_parameters) {
                switch (p.first) {
                case GL_TEXTURE_MIN_FILTER:
                    setTextureMinFilter(p.second);
                    break;
                case GL_TEXTURE_MAG_FILTER:
                    setTextureMagFilter(p.second);
                    break;
                case GL_TEXTURE_WRAP_S:
                    setTextureWrapS(p.second);
                    break;
                case GL_TEXTURE_WRAP_T:
                    setTextureWrapT(p.second);
                    break;
                case GL_TEXTURE_WRAP_R:
                    setTextureWrapR(p.second);
                    break;
                case GL_TEXTURE_COMPARE_MODE:
                    setTextureCompareMode(p.second);
                    break;
                case GL_TEXTURE_COMPARE_FUNC:
                    setTextureCompareFunc(p.second);
                    break;
                default:
                    throw BaseException("Sampler::Sampler - sampler id: " + m_id + " - parameter " +
                                        std::to_string(p.first) + " is wrong");
                }
            }

            for (const auto& p : layout.float_parameters) {
                switch (p.first) {
                case GL_TEXTURE_MIN_LOD:
                    setTextureMinLod(p.second);
                    break;
                case GL_TEXTURE_MAX_LOD:
                    setTextureMaxLod(p.second);
                    break;
                default:
                    throw BaseException("Sampler::Sampler - sampler id: " + m_id + " - parameter " +
                                        std::to_string(p.first) + " is wrong");
                }
            }

            setTextureBorderColor(layout.border_color);

            auto err = glGetError();
            if (err != GL_NO_ERROR) {
                throw BaseException(
                    "Sampler::Sampler - sampler id: " + m_id + " - OpenGL error " + std::to_string(err));
            }
        }

        Sampler(std::string id, std::vector<std::pair<GLenum, GLint>> const& int_params) : m_id(id) {
            glCreateSamplers(1, &m_name);

            for (const auto& p : int_params) {
                switch (p.first) {
                case GL_TEXTURE_MIN_FILTER:
                    setTextureMinFilter(p.second);
                    break;
                case GL_TEXTURE_MAG_FILTER:
                    setTextureMagFilter(p.second);
                    break;
                case GL_TEXTURE_WRAP_S:
                    setTextureWrapS(p.second);
                    break;
                case GL_TEXTURE_WRAP_T:
                    setTextureWrapT(p.second);
                    break;
                case GL_TEXTURE_WRAP_R:
                    setTextureWrapR(p.second);
                    break;
                case GL_TEXTURE_COMPARE_MODE:
                    setTextureCompareMode(p.second);
                    break;
                case GL_TEXTURE_COMPARE_FUNC:
                    setTextureCompareFunc(p.second);
                    break;
                default:
                    throw BaseException(
                        "Sampler::Sampler - sampler id: " + m_id + " - parameter " + std::to_string(p.first) + " is wrong");
                }
            }

            auto err = glGetError();
            if (err != GL_NO_ERROR)
            {
                throw BaseException("Sampler::Sampler - sampler id: " + m_id + " - OpenGL error " +
                                    std::to_string(err));
            }
        }

        Sampler(std::string id, std::vector<std::pair<GLenum, GLfloat>> const& float_params) : m_id(id) {
            glCreateSamplers(1, &m_name);

            for (const auto& p : float_params) {
                switch (p.first) {
                case GL_TEXTURE_MIN_LOD:
                    setTextureMinLod(p.second);
                    break;
                case GL_TEXTURE_MAX_LOD:
                    setTextureMaxLod(p.second);
                    break;
                default:
                    throw BaseException("Sampler::Sampler - sampler id: " + m_id + " - parameter " +
                                        std::to_string(p.first) + " is wrong");
                }
            }

            auto err = glGetError();
            if (err != GL_NO_ERROR)
            {
                throw BaseException("Sampler::Sampler - sampler id: " + m_id + " - OpenGL error " +
                                    std::to_string(err));
            }
        }

        ~Sampler() {
            glDeleteSamplers(1, &m_name);
        }

        Sampler(const Sampler&) = delete;
        Sampler(Sampler&& other) = delete;
        Sampler& operator=(const Sampler& rhs) = delete;
        Sampler& operator=(Sampler&& rhs) = delete;

        void bindSampler(GLuint tex_unit) const {
            glBindSampler(tex_unit, m_name);
        }

        std::string getId() const {
            return m_id;
        }

        GLuint getName() const {
            return m_name;
        }

        // TODO: ??
//#ifndef GLOWL_NO_ARB_BINDLESS_TEXTURE
//        GLuint64 getTextureHandle() const
//        {
//            return m_texture_handle;
//        }
//        GLuint64 getImageHandle(GLint level, GLboolean layered, GLint layer) const
//        {
//            return glGetImageHandleARB(m_name, level, layered, layer, m_internal_format);
//        }
//#endif

        GLint getTextureMinFilter() const {
            return m_texture_min_filter;
        }
        GLint getTextureMagFilter() const {
            return m_texture_mag_filter;
        }
        GLfloat getTextureMinLod() const {
            return m_texture_min_lod;
        }
        GLfloat getTextureMaxLod() const {
            return m_texture_max_lod;
        }
        GLint getTextureWrapS() const {
            return m_texture_wrap_s;
        }
        GLint getTextureWrapT() const {
            return m_texture_wrap_t;
        }
        GLint getTextureWrapR() const {
            return m_texture_wrap_r;
        }
        std::array<GLfloat, 4> getTextureBorderColor() const {
            return m_texture_border_color;
        }
        GLint getTextureCompareMode() const {
            return m_texture_compare_mode;
        }
        GLint getTextureCompareFunc() const {
            return m_texture_compare_func;
        }

        void setTextureMinFilter(GLint filter) {
            m_texture_min_filter = filter;
            glSamplerParameteri(m_name, GL_TEXTURE_MIN_FILTER, filter);
        }
        void setTextureMagFilter(GLint filter) {
            m_texture_mag_filter = filter;
            glSamplerParameteri(m_name, GL_TEXTURE_MAG_FILTER, filter);
        }
        void setTextureMinLod(GLfloat lod) {
            m_texture_min_lod = lod;
            glSamplerParameterf(m_name, GL_TEXTURE_MIN_LOD, lod);
        }
        void setTextureMaxLod(GLfloat lod) {
            m_texture_max_lod = lod;
            glSamplerParameterf(m_name, GL_TEXTURE_MAX_LOD, lod);
        }
        void setTextureWrapS(GLint s) {
            m_texture_wrap_s = s;
            glSamplerParameteri(m_name, GL_TEXTURE_WRAP_S, s);
        }
        void setTextureWrapT(GLint t) {
            m_texture_wrap_t = t;
            glSamplerParameteri(m_name, GL_TEXTURE_WRAP_T, t);
        }
        void setTextureWrapR(GLint r) {
            m_texture_wrap_r = r;
            glSamplerParameteri(m_name, GL_TEXTURE_WRAP_R, r);
        }
        void setTextureBorderColor(const std::array<GLfloat, 4>& col) {
            m_texture_border_color = col;
            glSamplerParameterfv(m_name, GL_TEXTURE_BORDER_COLOR, col.data());
        }
        void setTextureCompareMode(GLint mode) {
            m_texture_compare_mode = mode;
            glSamplerParameteri(m_name, GL_TEXTURE_COMPARE_MODE, mode);
        }
        void setTextureCompareFunc(GLint func) {
            m_texture_compare_func = func;
            glSamplerParameteri(m_name, GL_TEXTURE_COMPARE_FUNC, func);
        }

    private:
        std::string m_id; ///< Identifier set by application to help identifying samplers

        GLuint m_name; ///< OpenGL sampler name given by glCreateSampler
#ifndef GLOWL_NO_ARB_BINDLESS_TEXTURE
        GLuint64 m_sampler_handle; ///< Actual OpenGL sampler handle (used for bindless)
#endif

        GLint m_texture_min_filter;
        GLint m_texture_mag_filter;
        GLfloat m_texture_min_lod;
        GLfloat m_texture_max_lod;
        GLint m_texture_wrap_s;
        GLint m_texture_wrap_t;
        GLint m_texture_wrap_r;
        std::array<GLfloat, 4> m_texture_border_color;
        GLint m_texture_compare_mode;
        GLint m_texture_compare_func;
    };

} // namespace glowl

#endif // GLOWL_SAMPLER_HPP
