uniform vec2 scaling = vec2(1.0);
uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

uniform uint dimensionCount = 0;
uniform uint itemCount = 0;

uniform float pc_item_defaultDepth = 0.0;

uniform vec4 color = vec4(1.0, 0.0, 1.0, 1.0);
uniform float tfColorFactor = 1.0;
uniform int colorColumn = -1;
uniform vec2 margin = vec2(0.0, 0.0);
uniform float axisDistance = 0.0;
uniform float axisHeight = 0.0;
uniform float axisHalfTick = 2.0;
uniform uint numTicks = 3;
uniform int pickedAxis = -1;

uniform uint fragmentTestMask = 0;
uniform uint fragmentPassMask = 0;

uniform uint tick = 0;

uniform vec2 mouse = vec2(0, 0);
uniform float pickRadius = 0.1;

uniform vec2 mousePressed = vec2(0, 0);
uniform vec2 mouseReleased = vec2(0, 0);
