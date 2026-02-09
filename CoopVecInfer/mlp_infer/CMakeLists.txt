include(../donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project mlp_infer)
set(folder "MLP Infer")

donut_compile_shaders_all_platforms(
        TARGET ${project}_shaders
        PROJECT_NAME "MLP Infer"
        CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
        FOLDER ${folder}
        OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}
)

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine nvapi)
add_dependencies(${project} dx12-agility-sdk ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

target_compile_definitions(
        ${project} PRIVATE DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION}
        ${project} PRIVATE DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION=${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION}
)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()
