#ifndef MLTOOLING_CORE_ERROR_H
#define MLTOOLING_CORE_ERROR_H
#ifdef _WIN32
    #ifdef MLTOOLING_C_EXPORTS
        #define MLTOOLING_C __declspec(dllexport)
    #else
        #define MLTOOLING_C __declspec(dllimport)
    #endif
#else
    #define MLTOOLING_C __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

#define MLT_MAX_ERROR_MSG_LEN 50

typedef enum MLTerrorType
{
    MLT_ERROR_NONE,
    MLT_NULL_PTR,
    MLT_OUT_OF_MEMORY,
    MLT_INVALID_ALIGNMENT,
    MLT_OUT_OF_BOUNDS,
    MLT_TYPE_MISMATCH,
    MLT_RANK_MISMATCH,
    MLT_SHAPE_MISMATCH,
    MLT_INVALID_ARGUMENT,
    MLT_INTERNAL_ERROR
} MLTerrorType;

typedef struct {
    MLTerrorType type;
    char message[MLT_MAX_ERROR_MSG_LEN + 1];
} MLTerror;

static MLTerror mltErrorCreate(MLTerrorType type, const char* msg)
{
    MLTerror err = { .type = type };
    snprintf(err.message, sizeof(err.message), "%s", msg);

    return err;
}

#ifdef __cplusplus
}
#endif

#endif
