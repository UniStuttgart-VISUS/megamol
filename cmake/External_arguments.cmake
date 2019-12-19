#
# Set the default value for an argument.
#
function(_argument_default VARIABLE)
  if(args_${VARIABLE})
    set(${VARIABLE} "${args_${VARIABLE}}" PARENT_SCOPE)
  else()
    set(${VARIABLE} "${ARGN}" PARENT_SCOPE)
  endif()
endfunction(_argument_default)
