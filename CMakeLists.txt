cmake_minimum_required(VERSION 3.20)
project(LampAnalyser)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(OpenCV_DIR "C:/opencv/x64/vc17/lib")
endif()

find_package(Qt6 COMPONENTS Core Gui Widgets OpenGLWidgets Charts REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(UNIX)
    include_directories(/usr/local/include)
    include_directories(/usr/local/include/opencv4)
    link_directories(/opt/pylon/lib)
    link_directories(/usr/local/lib)
elseif(WIN32)
    include_directories(C:/opencv/include/opencv2)
    include_directories(C:/Qt/Qt-6.9.0/include)
    link_directories(C:/Qt/Qt-6.9.0/lib)
    link_directories(C:/opencv/x64/vc17/lib)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/mainwindow
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/mainwindow/mainwindow.cpp
    ${CMAKE_SOURCE_DIR}/mainwindow/visualwidget.cpp
    ${CMAKE_SOURCE_DIR}/mainwindow/mainwindow.h
    ${CMAKE_SOURCE_DIR}/mainwindow/mainwindow.ui
    ${CMAKE_SOURCE_DIR}/src/GLVisualisation.cpp
    ${CMAKE_SOURCE_DIR}/inc/GLVisualisation.h
    ${CMAKE_SOURCE_DIR}/src/ImageProcessor.cpp
    ${CMAKE_SOURCE_DIR}/inc/ImageProcessor.h
    ${CMAKE_SOURCE_DIR}/src/LampImageAnalyser.cpp
    ${CMAKE_SOURCE_DIR}/inc/LampImageAnalyser.h
)

add_executable(${PROJECT_NAME} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    WIN32_EXECUTABLE TRUE  # hides console on Windows
    MACOSX_BUNDLE TRUE     # makes a proper .app on macOS
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::OpenGLWidgets Qt6::Charts)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui        
        opencv_ximgproc
    )
elseif(WIN32)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS}
        opengl32
    )
endif()
