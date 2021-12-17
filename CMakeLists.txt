cmake_minimum_required(VERSION 3.15)
project(RegistrationGL)

set(CMAKE_CXX_STANDARD 14)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS}
)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(PCL REQUIRED COMPONENTS common io registration)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})

include_directories(3rd/Sophus)
include_directories(3rd/imgui 3rd/imgui/backends)
include_directories(3rd/ImGuizmo)

find_package(TBB COMPONENTS tbbmalloc tbbmalloc_proxy tbb_preview)

set(ImGui_Glut_src 3rd/imgui/imgui.cpp 3rd/imgui/imgui_demo.cpp 3rd/imgui/imgui_draw.cpp 3rd/imgui/imgui_widgets.cpp
        3rd/imgui/backends/imgui_impl_glut.cpp 3rd/imgui/backends/imgui_impl_opengl2.cpp)
set(ImGuizmo_src 3rd/ImGuizmo/ImGuizmo.cpp)

include_directories(GL 3rd 3rd/imgui 3rd/imgui/backends)
add_library(GL_Codes STATIC
        # wrapers
        GL/IndexBuffer.cpp
        GL/Renderer.cpp
        GL/Texture.cpp
        GL/VertexBuffer.cpp
        GL/Shader.cpp
        GL/VertexArray.cpp
        GL/VertexBufferLayout.cpp
        # imgui
        3rd/imgui/imgui.cpp 3rd/imgui/imgui_demo.cpp 3rd/imgui/imgui.cpp
        3rd/imgui/imgui_draw.cpp 3rd/imgui/imgui_widgets.cpp
        3rd/imgui/backends/imgui_impl_opengl3.cpp 3rd/imgui/backends/imgui_impl_glfw.cpp
        #ImViewGizmo
        3rd/ImGuizmo/ImGuizmo.cpp
)


find_package(fmt)

find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs


add_executable(slamGL main.cpp)

target_link_libraries(slamGL
         ${GLEW_LIBRARIES} ${CERES_LIBRARIES} ${OPENGL_LIBRARIES} glfw  fmt::fmt Eigen3::Eigen ${PCL_LIBRARIES} gtsam tbb Boost::program_options)
target_link_libraries(slamGL GL_Codes )

