#ifndef CORE_HPP
#define CORE_HPP

// Loads OpenGL, GLFW and provides convenience functions.

#pragma once

// OpenGL loader
#include "glad/glad.h"

// OpenGL framework
//#include "GLFW/glfw3.h"

#include <vector>

/// RAII class for initializing and terminating GLFW.
/// This is a singleton since GLFW should be initialized exactly once.
class GlfwInstance {
  public:
    /// Ensures that GLFW is initialized.
    static void init();

  private:
    /// Constructor initializes GLFW.
    GlfwInstance();

    /// Destructor terminates GLFW.
    ~GlfwInstance();
};

/// Compiles a shader.
/// @param shader Handle of the shader.
/// @param code The shader code to compile.
/// @throw std::runtime_error if compilation failed.
void CompileShader(GLuint shader, const char *file_path);

/// Links a shader program.
/// @param program Handle of the shader program to link.
/// @param shaders Handles of the shaders to attach before linking.
/// @throw std::runtime_error if linking failed.
void LinkProgram(GLuint program, const std::vector<GLuint> &shaders);

/// Interpolates linearly in (0, val1), (1, val2).
/// @param x The place to evaluate the interpolation at.
/// @param val1,val2 The values to interpolate.
/// @return The value of the linear interpolation at x.
template <class T> T lerp(float x, T val1, T val2) { return (1 - x) * val1 + x * val2; }

#endif // CORE_HPP
