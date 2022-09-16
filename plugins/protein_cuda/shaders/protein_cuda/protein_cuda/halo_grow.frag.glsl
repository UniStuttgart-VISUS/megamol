uniform sampler2D sourceTex;
uniform vec2 screenResInv;

#define FILTERSAMPLES 5
#define FILTEROFFSET  2

void main(void)
{
    vec4 cOut = vec4(0.0);
    vec4 src = texture2D(sourceTex, gl_TexCoord[0].xy);

    vec2 stepSize = screenResInv;

    for(int y = -FILTERSAMPLES-1; y < FILTERSAMPLES; ++y)
      for(int x = -FILTERSAMPLES-1; x < FILTERSAMPLES; ++x)
      {
        vec2 texPos = gl_TexCoord[0].xy + stepSize * vec2(x, y);
        vec4 sample = texture2D(sourceTex, texPos);
        vec3 col = abs(sample.rgb);
        float epsilon = 0.01;
        if( col.r > epsilon && col.g > epsilon && col.b > epsilon)
        {
          gl_FragColor = sample;
          return;
        }
      }
    discard;
}
