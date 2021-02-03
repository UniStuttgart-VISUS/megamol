# compositing_gl

The compositing_gl plugin offers a set of modules and helper classes for screen space effects and texture compositing operations for deferred renering.

# Modules

## SimpleRenderTarget

A `Renderer3DModule` that contains a Framebufferobject and binds it for use with chained rendering. The `Simple` variant features a surface color, normals and scalar depth render target.

## ScreenSpaceEffect

A module that computes screen space effects from given input render target textures. Required render targets depend on the selected effect, e.g. FXAA only requires a color image while SSAO requires normals, depth and camera information instead.

## TextureCombine

A module that combines two input texture on texel level. The output texture size is set from the first input texture. The output format currently is fixed to RGBA 16bit per channel. Currently supports addition and multiplication.

## DrawToScreen

A `Renderer3DModule` that draws a given texture to the output window, i.e. default framebuffer.