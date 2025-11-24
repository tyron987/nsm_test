AC_INIT([mnoz], [1.0], [twoj_email@example.com])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_PROG_CC

# Sprawdzenie biblioteki matematycznej
AC_CHECK_LIB([m], [sin], [], [
  AC_MSG_ERROR([libm not found])
])

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT

