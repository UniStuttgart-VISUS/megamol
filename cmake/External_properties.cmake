#
# Get the property of an external project.
#
# The value of the property will be stored in the variable <property> if
# the property was set for the given target. Otherwise, an error is issued.
#
# external_get_property(<target> <property>)
#
function(external_get_property TARGET PROPERTY)
  if(NOT DEFINED EXTERNAL_${TARGET}_${PROPERTY})
    message(FATAL_ERROR "Property ${PROPERTY} for target ${TARGET} is not defined")
  endif()

  set(${PROPERTY} ${EXTERNAL_${TARGET}_${PROPERTY}} PARENT_SCOPE)
endfunction()





#
# Try to get the property of an external project.
#
# The value of the property will be stored in the variable <property> if
# the property was set for the given target. Otherwise, it will be undefined.
#
# external_try_get_property(<target> <property>)
#
function(external_try_get_property TARGET PROPERTY)
  if(NOT DEFINED EXTERNAL_${TARGET}_${PROPERTY})
    unset(${PROPERTY} PARENT_SCOPE)
  else()
    set(${PROPERTY} ${EXTERNAL_${TARGET}_${PROPERTY}} PARENT_SCOPE)
  endif()
endfunction()





#
# Set the property of an external project.
#
# Set the property <property> for a given target to the desired value.
#
# external_set_property(<target> <property> <value>)
#
function(external_set_property TARGET PROPERTY VAR)
  set(EXTERNAL_${TARGET}_${PROPERTY} ${VAR} CACHE STRING "" FORCE)
  mark_as_advanced(EXTERNAL_${TARGET}_${PROPERTY})
endfunction()
