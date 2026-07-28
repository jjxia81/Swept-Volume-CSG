if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY qnzhou/cell-complex
    GIT_TAG jj_4dtet
    # GIT_TAG 185c45b7b26f4a44ed0d47491d2b85c0718b1b73
    # GIT_TAG main
)
