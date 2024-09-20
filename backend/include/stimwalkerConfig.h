#ifndef __STIMWALKER_CONFIG_H__
#define __STIMWALKER_CONFIG_H__

#define STIMWALKER_API // We kept this so if we want to convert to a shared
                       // library, we can do it easily
#ifdef _WIN32
#define _ENABLE_EXTENDED_ALIGNED_STORAGE
#endif

#define STIMWALKER_NAMESPACE stimwalker

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef _WIN32
#ifndef NAN
// static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
#define NAN (*(const float *)__nan)
#endif // NAN
#endif // _WIN32

#endif // __STIMWALKER_CONFIG_H__
