#ifndef MISC_H
#define MISC_H

#if _MSC_VER
#include <process.h>
#endif

int portable_getpid();
/*{
#ifdef _MSC_VER
  return _getpid();
#else
  return getpid();
#endif
}*/

#endif  //  end for MISC_H