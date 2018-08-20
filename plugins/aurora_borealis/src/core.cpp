#include "core.hpp"

#include <fstream>
#include <string>

void GlfwInstance::init() { static GlfwInstance instance; }

GlfwInstance::GlfwInstance() {
//    if (!glfwInit())
       // throw std::runtime_error("Failed to initialize GLFW.");
}

GlfwInstance::~GlfwInstance() { 
	//glfwTerminate(); 
}

/// Anonymous namspace. Intends that these functions are only visible in this file.
namespace {

/// Reads a file.
/// @param path Path to the file.
/// @return Content of the file as string.
/// @throw std::runtime_error if opening the file fails.
std::string ReadFile(const char *file_path) {
    std::ifstream file_stream(file_path);
    if (!file_stream.is_open())
        throw std::runtime_error(std::string("Failed to open ") + file_path);
    return std::string(std::istreambuf_iterator<char>(file_stream),
                       std::istreambuf_iterator<char>());
}

/// Gets the error log corresponding to a shader.
/// @param shader Handle of the shader to obtain the error log from.
/// @return Error log as string.
std::string GetShaderError(GLuint shader) {
    // get error log length
    GLint info_log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length == 0)
        return std::string();

    // copy error log into a buffer
    std::vector<GLchar> info_log(info_log_length);
    glGetShaderInfoLog(shader, info_log_length, nullptr, info_log.data());
    return std::string(info_log.data());
}

/// Gets the error log corresponding to a shader program.
/// @param shader Handle of the shader program to obtain the error log from.
/// @return Error log as string.
std::string GetProgramError(GLuint program) {
    // get error log length
    GLint info_log_length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length == 0)
        return std::string();

    // copy error log into a buffer
    std::vector<GLchar> info_log(info_log_length);
    glGetProgramInfoLog(program, info_log_length, nullptr, info_log.data());
    return std::string(info_log.data());
}

} // namespace

void CompileShader(GLuint shader, const char *file_path) {
    // compile
    std::string code = ReadFile(file_path);
    const char *p_code = code.c_str();
    glShaderSource(shader, 1, &p_code, nullptr);
    glCompileShader(shader);

    // check if compiling succeeded
    GLint compile_success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_success);
    if (compile_success == 0)
        throw std::runtime_error(std::string("Failed to compile shader \"") + file_path + "\".\n" +
                                 GetShaderError(shader));
}

void LinkProgram(GLuint program, const std::vector<GLuint> &shaders) {
    // attach shaders & link
    for (auto shader : shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);

    // check if it succeeded
    GLint link_success;
    glGetProgramiv(program, GL_LINK_STATUS, &link_success);
    if (link_success == 0)
        throw std::runtime_error("Failed to link shader program.\n" + GetProgramError(program));
}
