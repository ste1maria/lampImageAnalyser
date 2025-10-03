cmake_minimum_required(VERSION 3.20)
project(LampAnalyser)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    include_directories(C:/opencv/build/include)
    include_directories(C:/Qt/Qt-6.9.0/include)
    link_directories(C:/Qt/Qt-6.9.0/lib)
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
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::OpenGLWidgets Qt6::Charts)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
    )
elseif(WIN32)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS}
        opengl32
    )
endif()
