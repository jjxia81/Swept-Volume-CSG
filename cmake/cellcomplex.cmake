if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY duxingyi-charles/cell-complex
    GIT_TAG fa1f4772928f2f8a5c0f8eace86a9e625b1a697f
)
