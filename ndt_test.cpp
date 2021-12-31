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

#include "tbb/tbb.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>

#include  <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <boost/program_options.hpp>
#include <thread>



glm::vec2 clicked_point;
glm::vec3 view_translation{ 0,0,-30 };
float rot_x =0.0f;
float rot_y =0.0f;


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

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
            view_translation[1] -= 0.01 * d[1];
            view_translation[0] -= 0.01 * d[0];
        }
        clicked_point = p;
    }
}



struct grid_params{
    Eigen::Vector3d center;
    Eigen::Vector3d resolution;
    Eigen::Vector3i count;

    bool in(const Eigen::Vector3d & point)const{
        Eigen::Vector3d p_double= (point-center);
        if (p_double.x()>resolution.x()*count.x()) return false;
        if (p_double.y()>resolution.y()*count.y()) return false;
        if (p_double.z()>resolution.z()*count.z()) return false;

        if (p_double.x()<0) return false;
        if (p_double.y()<0) return false;
        if (p_double.z()<0) return false;

        return true;
    }
    int getIndex(const Eigen::Vector3d & point) const{
        Eigen::Vector3d p_double= (point-center);
        p_double.x() = std::floor(p_double.x()/resolution.x());
        p_double.y() = std::floor(p_double.y()/resolution.y());
        p_double.z() = std::floor(p_double.z()/resolution.z());
        int index = p_double.x() + p_double.y()* count.x() + p_double.z()*count.x()*count.y();
        assert (index < getCount());

        return index;
    }
    int getIndex(int x,int y,int z) const{
        int index = x + y* count.x() + z*count.x()*count.y();
        return index;
    }

    int getCount() const{
        return count.x()*count.y()*count.z();
    }
};

struct ndt{
    Eigen::Matrix3d cov{Eigen::Matrix3d::Zero()};
    Eigen::Vector3d mean;
    Eigen::Vector3d center;
    Eigen::Matrix4d ellipse;
    Eigen::Vector3f normal;
    int class_id{0};
    bool draw{false};
    void update(){
        ellipse = Eigen::Matrix4d::Identity();
        Eigen::LLT<Eigen::Matrix<double,3,3> > cholSolver(cov);
        ellipse.block<3,3>(0,0) = cholSolver.matrixL();
        ellipse.block<3,1>(0,3) = mean;
    }
};

int main(int argc, char **argv) {


    const grid_params gp{Eigen::Vector3d{-50, -30, -2}, Eigen::Vector3d{1.0, 1.0, 0.5}, Eigen::Vector3i{600, 600,10}};

    std::vector<ndt> ndt_array;
    ndt_array.resize(gp.getCount());
    std::vector<pcl::PointCloud<pcl::PointXYZI>> boxes;
    boxes.resize(gp.getCount());

    pcl::PointCloud<pcl::PointXYZI> pc;
    pcl::PointCloud<pcl::PointXYZI> scan_data;
    pcl::io::loadPCDFile("/media/michal/ext/p7p2.pcd", pc);
    pcl::io::loadPCDFile("/media/michal/ext/0.pcd", scan_data);

    //std::cout << gp.getIndex({0,0,0}) << std::endl;

    for (int i = 0; i < pc.size(); i++) {
        auto p = pc[i];
        if (gp.in({p.x, p.y, p.z})) {
            int index = gp.getIndex({p.x, p.y, p.z});
            boxes[index].push_back(p);
            ndt_array[index].draw = true;
        }
    }


    for (int i = 0; i < ndt_array.size(); i++) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(boxes[i], centroid);
        Eigen::Matrix3f cov;
        pcl::computeCovarianceMatrix(boxes[i], centroid, cov);

        Eigen::JacobiSVD<Eigen::Matrix3f> svd( cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

        ndt_array[i].cov = cov.cast<double>();
        ndt_array[i].mean = centroid.head<3>().cast<double>();
        ndt_array[i].normal = svd.matrixU().rightCols<1>();
        ndt_array[i].update();

    }
    boxes.clear();

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


    std::vector<float> sphere_data;

    const double pi = 3.141592;
    const double di = 0.02;
    const double dj = 0.04;
    const double du = di * 2 * pi;
    const double dv = dj * pi;

    for (double i = 0; i < 1.0; i += di) {//horizonal
        for (double j = 0; j < 1.0; j += dj)  //vertical
        {
            double u = i * 2 * pi;      //0     to  2pi
            double v = (j - 0.5) * pi;  //-pi/2 to pi/2
            const Eigen::Vector3d pp0(cos(v) * cos(u), cos(v) * sin(u), sin(v));
            const Eigen::Vector3d pp1(cos(v) * cos(u + du), cos(v) * sin(u + du), sin(v));
            const Eigen::Vector3d pp2(cos(v + dv) * cos(u + du), cos(v + dv) * sin(u + du), sin(v + dv));
            const Eigen::Vector3d pp3(cos(v + dv) * cos(u), cos(v + dv) * sin(u), sin(v + dv));

            sphere_data.push_back(0.2 * pp0.x());
            sphere_data.push_back(0.2 * pp0.y());
            sphere_data.push_back(0.2 * pp0.z());
            sphere_data.push_back(255);

            sphere_data.push_back(0.2 * pp1.x());
            sphere_data.push_back(0.2 * pp1.y());
            sphere_data.push_back(0.2 * pp1.z());
            sphere_data.push_back(255);

            sphere_data.push_back(0.2 * pp2.x());
            sphere_data.push_back(0.2 * pp2.y());
            sphere_data.push_back(0.2 * pp2.z());

            sphere_data.push_back(255);

            sphere_data.push_back(0.2 * pp3.x());
            sphere_data.push_back(0.2 * pp3.y());
            sphere_data.push_back(0.2 * pp3.z());
            sphere_data.push_back(255);

        }
    }
    VertexBufferLayout layout_pc;
    layout_pc.Push<float>(3);
    layout_pc.Push<float>(1);

    VertexArray va_sphere;
    VertexBuffer vb_sphere(sphere_data.data(), sphere_data.size() * sizeof(float));
    va_sphere.AddBuffer(vb_sphere, layout_pc);


    std::vector<float> cloud_data;
    for (int i = 0; i < scan_data.size(); i++)
    {
        const auto &p = scan_data[i];
        cloud_data.push_back(p.x);
        cloud_data.push_back(p.y);
        cloud_data.push_back(p.z);
        cloud_data.push_back(p.intensity);
    }


    VertexArray va_data_pointcloud;
    VertexBuffer vb_data_pointcloud(cloud_data.data(),cloud_data.size() * sizeof(float));
    va_data_pointcloud.AddBuffer(vb_data_pointcloud, layout_pc);

    Shader shader(shader_simple_v, shader_simple_f);
    Shader shader_pc(shader_pc_intensity_v, shader_pc_intensity_f);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    Eigen::Matrix4f imgizmo {Eigen::Matrix4f::Identity()};

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

        ImGuizmo::AllowAxisFlip(false);
        ImGuizmo::Manipulate(&model_rotation_3[0][0], &proj[0][0], ImGuizmo::TRANSLATE|ImGuizmo::ROTATE_Z, ImGuizmo::LOCAL, imgizmo.data(), NULL);

        glm::mat4 glm_gismo;
        Eigen::Map<Eigen::Matrix4f> map_glm_gismo(&glm_gismo[0][0]);
        map_glm_gismo = imgizmo;


        shader.Bind(); // bind shader to apply uniform
        shader.setUniformMat4f("u_MVP", proj * model_rotation_3);
        renderer.Draw(va, ib, shader, GL_LINES);

        shader_pc.Bind(); // bind shader to apply uniform
        shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * glm_gismo);
        shader_pc.setUniform4f("u_COLORPC", 1,1,1,1);
        renderer.DrawArray(va_data_pointcloud,shader_pc, GL_POINTS, cloud_data.size()/4);




        int overlap_count = 0;
        for (int i =0; i< scan_data.size(); i++)
        {
            Eigen::Vector4f pt = imgizmo * scan_data[i].getVector4fMap();
            auto ptd = pt.head<3>().cast<double>();
            if (gp.in(ptd)) {
                const int index = gp.getIndex(ptd);
                if (ndt_array[index].draw){
                    overlap_count++;
                }
            }
        }

        for (int i =0; i < gp.count.x();i++)
        {
            for (int j =0; j < gp.count.y();j++)
            {
                for (int k =0; k < gp.count.z();k++)
                {
                    const int index = gp.getIndex(i,j,k);
                    if ( !ndt_array[index].draw) continue;
                    glm::mat4 tr;
                    Eigen::Map<Eigen::Matrix4f> tr_d(&tr[0][0]);
                    tr_d = ndt_array[index].ellipse.cast<float>();
                    shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_3 * tr);
                    auto & n = ndt_array[index].normal;
                    if (n.z()>0.995) {
                        shader_pc.setUniform4f("u_COLORPC", 0, 0, 1, 1);
                    }else{
                        shader_pc.setUniform4f("u_COLORPC", 1, 0, 0, 1);
                    }
//                    shader_pc.setUniform4f("u_COLORPC", 0,0,(3+ndt_array[index].center.z())/5, 1);
                    renderer.DrawArray(va_sphere,shader_pc, GL_LINE_LOOP, sphere_data.size()/4);
                }
            }
        }


        ImGui::Begin("SLAM Demo");
        ImGui::Text("count = %d", overlap_count);
        ImGui::End();

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
