#include "stdafx.h"
#include "flowvis/shader.h"

#include "glad/glad.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        namespace utility
        {
            GLuint make_shader(const std::string& shader, GLenum type)
            {
                const GLchar* vertex_shader_ptr = shader.c_str();
                const GLint vertex_shader_length = static_cast<GLint>(shader.length());

                GLuint shader_handle = glCreateShader(type);
                glShaderSource(shader_handle, 1, &vertex_shader_ptr, &vertex_shader_length);
                glCompileShader(shader_handle);

                // Check compile status
                GLint compile_status;
                glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_status);

                if (compile_status == GL_FALSE)
                {
                    int info_log_length = 0;
                    glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &info_log_length);

                    if (info_log_length > 1)
                    {
                        int chars_written = 0;
                        std::vector<GLchar> info_log(info_log_length);

                        glGetShaderInfoLog(shader_handle, info_log_length, &chars_written, info_log.data());

                        throw std::runtime_error(info_log.data());
                    }
                    else
                    {
                        throw std::runtime_error("Unknown shader compile error");
                    }
                }

                return shader_handle;
            }

            GLuint make_program(const std::vector<GLuint>& shader_handles)
            {
                GLuint program_handle = glCreateProgram();

                for (const auto shader_handle : shader_handles)
                {
                    glAttachShader(program_handle, shader_handle);
                }

                glLinkProgram(program_handle);
                glUseProgram(0);

                // Check link status
                GLint link_status;
                glGetProgramiv(program_handle, GL_LINK_STATUS, &link_status);
                
                if (link_status == GL_FALSE)
                {
                    int info_log_length = 0;
                    glGetProgramiv(program_handle, GL_INFO_LOG_LENGTH, &info_log_length);

                    if (info_log_length > 1)
                    {
                        int chars_written = 0;
                        std::vector<GLchar> info_log(info_log_length);

                        glGetProgramInfoLog(program_handle, info_log_length, &chars_written, info_log.data());

                        throw std::runtime_error(info_log.data());
                    }
                    else
                    {
                        throw std::runtime_error("Unknown shader compile error");
                    }
                }

                return program_handle;
            }
        }
    }
}