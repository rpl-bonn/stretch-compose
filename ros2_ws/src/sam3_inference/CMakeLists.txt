cmake_minimum_required(VERSION 3.8)
project(sam3_inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BBox.msg"
  "msg/Detection.msg"
  "srv/InferenceSam3.srv"
  "srv/SceneInferenceSam3.srv"
  DEPENDENCIES sensor_msgs
)

install(PROGRAMS
  src/service/sam3_inference_service.py
  src/service/sam3_scenegraph_service.py
  src/client/sam3_inference_client.py
  src/client/sam3_scenegraph_client.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  src
  src/utils
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
