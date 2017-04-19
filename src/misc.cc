#include "misc.h"
#ifndef _MSC_VER
# include <sys/types.h>
# include <unistd.h>
#endif

int portable_getpid() {
#ifdef _MSC_VER  
  return _getpid();
#else
  return getpid();
#endif
}
