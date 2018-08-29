#ifndef GLSLProgram_h
#define GLSLProgram_h

#include "vislib/graphics/gl/IncludeAllGL.h"
//	OpenGL Math Library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

#include <string>
//#include <iostream>

/**
 * \class GLSLProgram
 *
 * \brief Encapsulates shader program functionality. Possibly somewhat outdated.
 *
 * \author Michael Becher
 */
class GLSLProgram {
private:
    GLuint m_handle;
    bool m_link_status;
    bool m_compute_shader;
    std::string m_shaderlog;

    // TODO: this is simply a hotfix solution
    std::string m_id;

    GLuint getUniformLocation(const char* name);

public:
    GLSLProgram();
    ~GLSLProgram();

    /* Deleted copy constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    GLSLProgram(const GLSLProgram& cpy) = delete;

    GLSLProgram(GLSLProgram&& other) = delete;

    GLSLProgram& operator=(const GLSLProgram& rhs) = delete;
    GLSLProgram& operator=(GLSLProgram&& rhs) = delete;

    void init();
    bool compileShaderFromString(const std::string* const source, GLenum shaderType);
    bool link();
    bool use();
    bool dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    const std::string& getLog();
    GLuint getHandle();
    bool isLinked();
    bool isComputeShader();
    void bindAttribLocation(GLuint location, const char* name);
    void bindFragDataLocation(GLuint location, const char* name);

    void setUniform(const char* name, const glm::vec2& v);
    void setUniform(const char* name, const glm::ivec2& v);
    void setUniform(const char* name, const glm::ivec3& v);
    void setUniform(const char* name, const glm::ivec4& v);
    void setUniform(const char* name, const glm::vec3& v);
    void setUniform(const char* name, const glm::vec4& v);
    void setUniform(const char* name, const glm::mat4& m);
    void setUniform(const char* name, const glm::mat3& m);
    void setUniform(const char* name, int i);
    void setUniform(const char* name, unsigned int i);
    void setUniform(const char* name, float f);
    void setUniform(const char* name, bool b);
    void printActiveUniforms();
    void printActiveAttributes();

    void setId(const std::string& id);
    std::string getId();
};

#endif // !GLSLProgram_hpp
