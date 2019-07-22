/*
 * GLSLProgram.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_GLSLPROGRAM_H_INCLUDED
#define MEGAMOLCORE_GLSLPROGRAM_H_INCLUDED

#include "vislib/graphics/gl/IncludeAllGL.h"
//	OpenGL Math Library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

#include <string>

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class GLSLProgram
 *
 * @brief Encapsulates shader program functionality. Possibly somewhat outdated.
 *
 * @author Michael Becher
 */
class GLSLProgram {
public:
    /**
     * Constructor
     */
    GLSLProgram();

    /**
     * Destructor
     */
    virtual ~GLSLProgram();

    /* Deleted copy constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    GLSLProgram(const GLSLProgram& cpy) = delete;

    /* Deleted move constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    GLSLProgram(GLSLProgram&& other) = delete;

    /* Deleted assignment operator (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    GLSLProgram& operator=(const GLSLProgram& rhs) = delete;

    /* Deleted move assignment (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    GLSLProgram& operator=(GLSLProgram&& rhs) = delete;

    /**
     * Initializes the program
     */
    void init();

    /**
     * Compiles a shader represented as string
     *
     * @param source
     * @param shaderType
     * @return
     */
    bool compileShaderFromString(const std::string* const source, GLenum shaderType);

    /**
     * Links the shader program
     *
     * @return True on success, false otherwise
     */
    bool link();

    /**
     * Enables usage of the shader program. This only works if compiling and linking was successful previously
     *
     * @return True on success, false otherwise
     */
    bool use();

    /**
     * Dispatches a compute operation that works only if this is a compute shader program
     *
     * @param num_groups_x Number of work groups in x direction
     * @param num_groups_y Number of work groups in y direction
     * @param num_groups_z Number of work groups in z direction
     * @return True on success, false otherwise
     */
    bool dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);

    /**
     * Get the log of the shader program
     *
     * @return The log of the program as string
     */
    const std::string& getLog();

    /**
     * Get the handle of the shader program
     *
     * @return The handle of the program
     */
    GLuint getHandle();

    /**
     * Returns whether this program was successfully linked
     *
     * @return True if linking was successful, false otherwise
     */
    bool isLinked();

    /**
     * Returns whether this program represents a compute shader
     *
     * @return True if this program allows for compute operations, false otherwise
     */
    bool isComputeShader();

    /**
     * Binds a named attribute to a location
     *
     * @param location The id of teh location
     * @param name The name of the attribute
     */
    void bindAttribLocation(GLuint location, const char* name);

    /**
     * Bind a named attribute to a fragment shader output location
     *
     * @param location The id of the location
     * @param name The name of the attribute
     */
    void bindFragDataLocation(GLuint location, const char* name);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::vec2& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::ivec2& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::ivec3& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::ivec4& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::vec3& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param v The value to set
     */
    void setUniform(const char* name, const glm::vec4& v);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param m The value to set
     */
    void setUniform(const char* name, const glm::mat4& m);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param m The value to set
     */
    void setUniform(const char* name, const glm::mat3& m);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param i The value to set
     */
    void setUniform(const char* name, int i);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param i The value to set
     */
    void setUniform(const char* name, unsigned int i);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param f The value to set
     */
    void setUniform(const char* name, float f);

    /**
     * Sets an uniform value of the shader program
     *
     * @param name The name of the uniform
     * @param b The value to set
     */
    void setUniform(const char* name, bool b);

    /**
     * Returns a comma-seperated list of all active uniforms
     *
     * @param Uniform list as string
     */
    std::string getActiveUniforms();

    /**
     * Returns a comma-seperated list of all active attributes
     *
     * @return Attribute list as string
     */
    std::string getActiveAttributes();

    /**
     * Sets the id value
     *
     * @param id The new id value
     */
    void setId(const std::string& id);

    /**
     * Returns the id of the program
     *
     * @return The id of the program
     */
    std::string getId();

private:
    /**
     * Returns the uniform location of an uniform with a given name
     *
     * @param name The name of the uniform
     * @return The location of the uniform within this program
     */
    GLuint getUniformLocation(const char* name);

    /** The handle of the shader program */
    GLuint m_handle;

    /** The link status of the shader program */
    bool m_link_status;

    /** Flag determining whether this is a compute shader or not */
    bool m_compute_shader;

    /** The compilation log the shaders */
    std::string m_shaderlog;

    // TODO: this is simply a hotfix solution
    std::string m_id;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_GLSLPROGRAM_H_INCLUDED
