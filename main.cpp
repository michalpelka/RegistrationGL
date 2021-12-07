#include "../GL/glwrapper.h"
#include "ImGuizmo/ImGuizmo.h"
//#include "utils.h"
#include <memory>
#include <sophus/se3.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <sophus/se3.hpp>

#include "costfun.h"
#include "struct.h"
#include "utils.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

glm::vec2 clicked_point;
glm::vec3 view_translation{ 0,0,-30 };
float rot_x =0.0f;
float rot_y =0.0f;


void cursor_calback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if(!io.WantCaptureMouse) {
        const glm::vec2 p{-xpos, ypos};
        const auto d = clicked_point - p;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            rot_x += 0.01 * d[1];
            rot_y += 0.01 * d[0];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            view_translation[2] += 0.02 * d[1];
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS) {
            view_translation[1] += 0.01 * d[1];
            view_translation[0] -= 0.01 * d[0];
        }
        clicked_point = p;
    }
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

std::vector<Eigen::Matrix4d> trajectory;
std::vector<std::unique_ptr<structs::KeyFrame>> keyframes;

int main(int argc, char **argv) {


    GLFWwindow *window;
    const char *glsl_version = "#version 130";
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(960, 540, "rgbd_demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_calback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) { return -1; }

    GLCall(glClearColor(0.4, 0.4, 0.4, 1));

    Renderer renderer;

    VertexBufferLayout layout;
    layout.Push<float>(3);
    layout.Push<float>(3);

    VertexArray va;
    VertexBuffer vb(gl_primitives::coordinate_system_vertex.data(),
                    gl_primitives::coordinate_system_vertex.size() * sizeof(float));
    va.AddBuffer(vb, layout);
    IndexBuffer ib(gl_primitives::coordinate_system_indices.data(), gl_primitives::coordinate_system_indices.size());

    Shader shader(shader_simple_v, shader_simple_f);
    Shader shader_pc(shader_pc_intensity_v, shader_pc_intensity_f);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);



//    VertexBufferLayout layoutPc;
//    layoutPc.Push<float>(3);
//    layoutPc.Push<float>(1); // ts
//    layoutPc.Push<float>(1); // intensity
//    std::vector<float> draw_buffer_vertices_pc1 = my_utils::loadTXTCloud("/home/michal/kopalnia_ws/src/m3d_export/data/029.txt");
//    VertexArray va_points1;
//    VertexBuffer vb_points1(draw_buffer_vertices_pc1.data(), draw_buffer_vertices_pc1.size() * sizeof(float));
//    va_points1.AddBuffer(vb_points1, layout);
//    auto fns = my_utils::glob("/home/michal/kopalnia_ws/src/m3d_export/data/*.mat");

    // load matrix
    std::vector<structs::edge> Edges;
    std::vector<std::string> fns = {};
    for (int i =0; i < 398; i+=3) {
        char data[128];
        snprintf(data,128, "%04d",i);
        fns.push_back("/home/michal/scans/cloud_"+std::string(data));
    }


    double off_x = 0;
    double off_y = 0;

    for (const auto fn : fns){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(fn+".pcd",*cloud);
        trajectory.push_back(my_utils::loadMat(fn+".txt"));
        if (off_x == 0)
        {
            off_x = trajectory.front()(0,3);
            off_y = trajectory.front()(1,3);
        }
        trajectory.back()(0,3) -= off_x;
        trajectory.back()(1,3) -= off_y;

        keyframes.emplace_back(std::make_unique<structs::KeyFrame>(cloud,  trajectory.back()));

    }
    const auto initial_poses = trajectory;

//    VertexBufferLayout layoutPc;
//    layoutPc.Push<float>(3);
//    layoutPc.Push<float>(1); // intensity
//    std::vector<float> draw_buffer_vertices_pc1 {1,0,0,1,
//                                                 2,0,0,1,
//                                                 3,0,0,1,
//                                                 4,0,0,1};
//    VertexArray va_points1;
//    VertexBuffer vb_points1(draw_buffer_vertices_pc1.data(), draw_buffer_vertices_pc1.size() * sizeof(float));
//    va_points1.AddBuffer(vb_points1, layout);


    Eigen::Matrix4f imgizmo {Eigen::Matrix4f::Identity()};
    int im_edited_frame = 113;
    int im_edited_frame_old = -1;

    while (!glfwWindowShouldClose(window)) {


        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::BeginFrame();
        ImGuizmo::Enable(true);
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

        /// OpenGL drawing
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glm::mat4 scale = glm::mat4(0.1f);
        glm::mat4 proj = glm::perspective(30.f, 1.0f*width/height, 0.1f, 1000.0f);
        glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
        glm::mat4 model_rotation_1 = glm::rotate(model_translate, rot_x, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 model_rotation_2 = glm::rotate(model_rotation_1, rot_y, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 model_rotation_3 = glm::rotate(model_rotation_2, (float)(0.5f*M_PI), glm::vec3(-1.0f, 0.0f, 0.0f));


        if (im_edited_frame >=0 && im_edited_frame < keyframes.size() ){
            if (im_edited_frame!= im_edited_frame_old){
                imgizmo = keyframes[im_edited_frame]->mat.cast<float>();
            }
            ImGuizmo::AllowAxisFlip(false);
            ImGuizmo::Manipulate(&model_rotation_3[0][0], &proj[0][0], ImGuizmo::TRANSLATE_X|ImGuizmo::TRANSLATE_Y|ImGuizmo::ROTATE_Z, ImGuizmo::LOCAL, imgizmo.data(), NULL);

        }

        im_edited_frame_old = im_edited_frame;

        shader.Bind(); // bind shader to apply uniform
        shader.setUniformMat4f("u_MVP", proj * model_rotation_3);
        renderer.Draw(va, ib, shader, GL_LINES);
//        // draw reference frame
        for (const auto &k : keyframes) {
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();
            shader.setUniformMat4f("u_MVP", proj * model_rotation_3 * local * scale);
            renderer.Draw(va, ib, shader, GL_LINES);
        }
        GLCall(glPointSize(1));
        for ( int i =0; i < keyframes.size(); i++) {
            const auto &k = keyframes[i];
            shader_pc.Bind();
            glm::mat4 local;
            Eigen::Map<Eigen::Matrix4f> map_local(&local[0][0]);
            map_local = k->mat.cast<float>();

            if (i == im_edited_frame ){
                shader_pc.setUniform4f("u_COLORPC", 1,1,1,1);
            }else{

                shader_pc.setUniform4f("u_COLORPC", 1,0,0,1);
            }
            shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * local );
            renderer.DrawArray(k->va, shader_pc, GL_POINTS, k->cloud->size());
        }

//
//        shader_pc.Bind();
//        GLCall(glPointSize(1));
//        shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_2);
//        shader_pc.setUniform4f("u_COLORPC", 1,0,0,1);
        //renderer.DrawArray(va_points1, shader_pc, GL_POINTS, draw_buffer_vertices_pc1.size()/6);
        //renderer.Draw(va, ib, shader, GL_LINES);
        // draw keyframes


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ImGui::Begin("SLAM Demo");
        ImGui::InputInt("Edited_frame",&im_edited_frame);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        if(ImGui::Button("reset view")){
            rot_x =0.0f;
            rot_y =0.0f;
            view_translation = glm::vec3{ 0,0,-30 };
            view_translation = glm::vec3{ 0,0,-30 };
        }
        if (ImGui::Button("ceres-relax")){
            std::vector<Sophus::SE3d> parameters;
            ceres::Problem problem;
            for (int i =0; i < im_edited_frame+1; i++){
                parameters.push_back(Sophus::SE3d( orthogonize(keyframes[i]->mat)));
            }
            for (int i =0; i < im_edited_frame+1; i++){
                problem.AddParameterBlock(parameters[i].data(), Sophus::SE3d::num_parameters, new LocalParameterizationSE3());
                //problem.SetParameterBlockConstant(parameters[i].data());
            }
            // frezee firts and edited frame
            problem.SetParameterBlockConstant(parameters.front().data());
            problem.SetParameterBlockConstant(parameters.back().data());

            for (int i =1; i < im_edited_frame+1; i++){
                Sophus::SE3d odom1(orthogonize(trajectory[i-1]));
                Sophus::SE3d odom2(orthogonize(trajectory[i]));
                auto c = RelativePose::Create(odom1,odom2);
                ceres::LossFunction *loss = nullptr;
                problem.AddResidualBlock(c,loss,parameters[i-1].data(),parameters[i].data());
            }
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 100;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";
            for (int i =0; i < im_edited_frame+1; i++){
                keyframes[i]->mat = parameters[i].matrix();
            }

        }
        if (ImGui::Button("gtsam-relax")){
            using namespace std;
            using namespace gtsam;
            NonlinearFactorGraph graph;

            auto priorModel = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());

            graph.add(PriorFactor<Pose3>(0, Pose3(keyframes[0]->mat), priorModel));
            graph.add(PriorFactor<Pose3>(im_edited_frame, Pose3(keyframes[im_edited_frame]->mat), priorModel));

            for (int i =1; i < im_edited_frame+1; i++){
                auto odometryNoise = noiseModel::Diagonal::Variances(
                        (Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
                Eigen::Matrix4d update = trajectory[i-1].inverse() * trajectory[i];
                graph.emplace_shared<BetweenFactor<Pose3> >(i-1, i, Pose3(orthogonize(update)), odometryNoise);
            }
            graph.print("\nFactor Graph:\n");  // print
            Values initial;
            for (int i =0; i < im_edited_frame+1; i++){
                initial.insert(i, Pose3(orthogonize(keyframes[i]->mat)));
            }
            Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
            result.print("Final Result:\n");
            for (int i =0; i < im_edited_frame+1; i++){
                auto v =  result.at<Pose3>(i);
                keyframes[i]->mat =v.matrix();
            }
        }
        if (ImGuizmo::IsOver()) {
        //if (ImGui::Button("t")) {
            if (im_edited_frame >= 0 && im_edited_frame < keyframes.size()) {
                const Eigen::Matrix4d before_inv{keyframes[im_edited_frame]->mat.inverse()};
                keyframes[im_edited_frame]->mat = imgizmo.cast<double>();
                std::cout << "aaa" << std::endl;
                std::cout << keyframes[im_edited_frame]->mat * before_inv << std::endl;
                for (int i = im_edited_frame + 1; i < keyframes.size(); i++) {
                    keyframes[i]->mat = keyframes[im_edited_frame]->mat * before_inv * keyframes[i]->mat;
                }
            }
        //}
        }
        if (ImGui::Button("reset")) {
            for (int i = 0; i < keyframes.size(); i++) {
                keyframes[i]->mat = trajectory[i].matrix();
            }
        }
        ImGui::End();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();

    }


    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;

}