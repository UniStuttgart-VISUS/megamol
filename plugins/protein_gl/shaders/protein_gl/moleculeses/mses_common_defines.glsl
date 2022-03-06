uniform vec4 viewAttr;
uniform vec3 zValues;
uniform vec3 fogCol;
uniform float alpha = 0.5;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 viewInverse;
uniform mat4 mvp;
uniform mat4 mvpinverse;
uniform mat4 mvptransposed;

in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;