#include "stdafx.h"
#include "mmcore/utility/gl/GLSLProgram.h"

#include <iostream>

using namespace megamol::core::utility::gl;

GLSLProgram::GLSLProgram() : m_link_status(false), m_compute_shader(false) {}

GLSLProgram::~GLSLProgram() { glDeleteProgram(m_handle); }


GLuint GLSLProgram::getUniformLocation(const char* name) { return glGetUniformLocation(m_handle, name); }

void GLSLProgram::init() { m_handle = glCreateProgram(); }

bool GLSLProgram::compileShaderFromString(const std::string* const source, GLenum shaderType) {
    /* Check if the source is empty */
    if (source->empty()) {
        m_shaderlog = "No shader source.";
        return false;
    }

    /* Create shader object */
    const GLchar* c_source = source->c_str();
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &c_source, NULL);

    if (shaderType == GL_COMPUTE_SHADER) m_compute_shader = true;

    /* Compile shader */
    glCompileShader(shader);

    /* Check for errors */
    GLint compile_ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_ok);
    if (compile_ok == GL_FALSE) {
        GLint logLen = 0;
        m_shaderlog = "";
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            char* log = new char[logLen];
            GLsizei written;
            glGetShaderInfoLog(shader, logLen, &written, log);
            m_shaderlog = log;
            delete[] log;
        }

        glDeleteShader(shader);
        return false;
    }

    //	GLint logLen = 0;
    //	m_shaderlog = "";
    //	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
    //	if(logLen > 0)
    //	{
    //		char *log = new char[logLen];
    //		GLsizei written;
    //		glGetShaderInfoLog(shader, logLen, &written, log);
    //		m_shaderlog = log;
    //		delete [] log;
    //	}

    /* Attach shader to program */
    glAttachShader(m_handle, shader);
    /* Flag shader program for deletion.
     * It will only be actually deleted after the program is deleted. (See destructor for program deletion.
     */
    glDeleteShader(shader);

    return true;
}

bool GLSLProgram::link() {
    if (m_link_status) return true;
    glLinkProgram(m_handle);

    GLint status = GL_FALSE;
    glGetProgramiv(m_handle, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint logLen = 0;
        // m_shaderlog = "";
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            char* log = new char[logLen];
            GLsizei written;
            glGetProgramInfoLog(m_handle, logLen, &written, log);
            m_shaderlog.append(log);
            delete[] log;
        }
        return false;
    }

    //	GLint logLen = 0;
    //	m_shaderlog = "";
    //	glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLen);
    //	if(logLen > 0)
    //	{
    //		char *log = new char[logLen];
    //		GLsizei written;
    //		glGetProgramInfoLog(m_handle, logLen, &written, log);
    //		m_shaderlog = log;
    //		delete [] log;
    //	}
    //	return false;

    m_link_status = true;
    return m_link_status;
}

bool GLSLProgram::use() {
    if (!m_link_status) return false;

    glUseProgram(m_handle);

    return true;
}

bool GLSLProgram::dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    GLuint current_prgm;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&current_prgm);

    if ((current_prgm != m_handle) || !m_compute_shader) return false;

    glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);

    return true;
}

const std::string& GLSLProgram::getLog() { return m_shaderlog; }

GLuint GLSLProgram::getHandle() { return m_handle; }

bool GLSLProgram::isLinked() { return m_link_status; }

void GLSLProgram::bindAttribLocation(GLuint location, const char* name) {
    glBindAttribLocation(m_handle, location, name);
}

void GLSLProgram::bindFragDataLocation(GLuint location, const char* name) {
    glBindFragDataLocation(m_handle, location, name);
}

void GLSLProgram::setUniform(const char* name, const glm::vec2& v) {
    glUniform2fv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::ivec2& v) {
    glUniform2iv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::ivec3& v) {
    glUniform3iv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::ivec4& v) {
    glUniform4iv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::vec3& v) {
    glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::vec4& v) {
    glUniform4fv(getUniformLocation(name), 1, glm::value_ptr(v));
}

void GLSLProgram::setUniform(const char* name, const glm::mat4& m) {
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(m));
}

void GLSLProgram::setUniform(const char* name, const glm::mat3& m) {
    glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(m));
}

void GLSLProgram::setUniform(const char* name, int i) { glUniform1i(getUniformLocation(name), i); }

void GLSLProgram::setUniform(const char* name, unsigned int i) { glUniform1ui(getUniformLocation(name), i); }

void GLSLProgram::setUniform(const char* name, float f) { glUniform1f(getUniformLocation(name), f); }

void GLSLProgram::setUniform(const char* name, bool b) { glUniform1i(getUniformLocation(name), b); }

void GLSLProgram::printActiveUniforms() {
    GLint maxLength, nUniforms;
    glGetProgramiv(m_handle, GL_ACTIVE_UNIFORMS, &nUniforms);
    glGetProgramiv(m_handle, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxLength);

    GLchar* attributeName = (GLchar*)new char[maxLength];

    GLint size, location;
    GLsizei written;
    GLenum type;

    for (int i = 0; i < nUniforms; i++) {
        glGetActiveUniform(m_handle, i, maxLength, &written, &size, &type, attributeName);
        location = glGetUniformLocation(m_handle, attributeName);
        std::cout << location << " - " << attributeName << "\n";
    }
    delete[] attributeName;
}

void GLSLProgram::printActiveAttributes() {
    GLint maxLength, nAttributes;
    glGetProgramiv(m_handle, GL_ACTIVE_ATTRIBUTES, &nAttributes);
    glGetProgramiv(m_handle, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxLength);

    GLchar* attributeName = (GLchar*)new char[maxLength];

    GLint written, size, location;
    GLenum type;

    for (int i = 0; i < nAttributes; i++) {
        glGetActiveAttrib(m_handle, i, maxLength, &written, &size, &type, attributeName);
        location = glGetAttribLocation(m_handle, attributeName);
        std::cout << location << " - " << attributeName << "\n";
    }
    delete[] attributeName;
}

void GLSLProgram::setId(const std::string& id) { m_id = id; }

std::string GLSLProgram::getId() { return m_id; }
