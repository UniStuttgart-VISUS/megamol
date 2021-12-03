float lambert(vec3 normal, vec3 light_dir)
{
    return clamp(dot(normal,light_dir),0.0,1.0);
}
