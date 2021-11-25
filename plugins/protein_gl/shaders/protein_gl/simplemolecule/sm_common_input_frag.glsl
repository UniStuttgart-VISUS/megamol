uniform vec4 viewAttr;

uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;

uniform mat4 MVinv;
uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;
uniform mat4 NormalM;

in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;
in vec4 move_color;
in vec4 move_color2;

uniform vec2 planes;
uniform vec3 clipPlaneDir;
uniform vec3 clipPlaneBase;

uniform bool applyFiltering;
uniform bool useClipPlane;
