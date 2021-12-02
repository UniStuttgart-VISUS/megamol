#ifndef CORE_BITFLAGS_INC_GLSL
#define CORE_BITFLAGS_INC_GLSL

const uint FLAG_ENABLED = 1u << 0;
const uint FLAG_FILTERED = 1u << 1;
const uint FLAG_SELECTED = 1u << 2;
const uint FLAG_SOFTSELECTED = 1u << 3;

bool bitflag_test(uint flags, uint test, uint pass)
{
    return (flags & test) == pass;
}

void bitflag_set(inout uint flags, uint flag, bool enabled)
{
    if (enabled)
    {
        flags |= flag;
    }
    else
    {
        flags &= ~flag;
    }
}

void bitflag_flip(inout uint flags, uint flag)
{
    flags ^= flag;
}

bool bitflag_isVisible(uint flags)
{
    return bitflag_test(flags, FLAG_ENABLED | FLAG_FILTERED, FLAG_ENABLED);
}

bool bitflag_isVisibleSelected(uint flags)
{
    return bitflag_test(flags, FLAG_ENABLED | FLAG_FILTERED | FLAG_SELECTED, FLAG_ENABLED | FLAG_SELECTED);
}

#endif // CORE_BITFLAGS_INC_GLSL
