#include "misc.h"

int portable_getpid() {
#ifdef _MSC_VER  
  return _getpid();
#else
  return getpid();
#endif
}