uniform vec4 viewAttr;

uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

uniform mat4 MVinv;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;
uniform mat4 NormalM;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out vec4 move_color;
out vec4 move_color2;

uniform vec2 planes;
uniform vec3 clipPlaneDir;
uniform vec3 clipPlaneBase;

uniform bool applyFiltering;
uniform bool useClipPlane;

layout (location = 0) in vec4 vert_position;
layout (location = 1) in vec3 vert_color;
layout (location = 2) in vec2 cyl_params;
layout (location = 3) in vec4 cyl_quat;
layout (location = 4) in vec3 cyl_color1;
layout (location = 5) in vec3 cyl_color2;
layout (location = 6) in int vert_filter;
