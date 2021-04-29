uniform uint instanceOffset = 0;

uint getInstanceID()
{
    return gl_InstanceID + instanceOffset;
}
