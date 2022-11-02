#define GLEW_STATIC
#include <GL/glwrapper.h>
#include "utils.h"
#include <memory>
glm::vec2 clicked_point;
float rot_x =0.0f;
float rot_y =0.0f;
bool drawing_buffer_dirty = true;
glm::vec3 view_translation{ 0,0,-30 };

void cursor_calback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if(!io.WantCaptureMouse) {
        const glm::vec2 p{-xpos, ypos};
        const auto d = clicked_point - p;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            rot_x += 0.01f * d[1];
            rot_y += 0.01f * d[0];
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


std::shared_ptr<float[]> pclToBuffer(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int &len, float scale)
{
    const int stride = 4;
    len = stride*cloud->size();
    std::shared_ptr<float[]> data (new float[len]);
    for (int i = 0; i < cloud->size();i++){
        data[stride*i+0] = scale*(*cloud)[i].x;
        data[stride*i+1] = scale*(*cloud)[i].y;
        data[stride*i+2] = scale*(*cloud)[i].z;
        data[stride*i+3] = (*cloud)[i].intensity;
    }
    return data;
}

void writePGM(std::string fn, const std::vector<u_int8_t>&data, int h, int w){
    FILE* pgmimg;
    pgmimg = fopen(fn.c_str(), "wb");

    // Writing Magic Number to the File
    fprintf(pgmimg, "P2\n");

    // Writing Width and Height
    fprintf(pgmimg, "%d %d\n", w, h);

    // Writing the maximum gray value
    fprintf(pgmimg, "255\n");
    int count = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            auto temp = data.at(y*w+x);
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    float imgui_depth {0};
    // load matrix

    GLFWwindow *window;
    const char *glsl_version = "#version 130";
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(960, 540, "demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_calback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) { return -1; }

    GLCall(glClearColor(0.4f, 0.4f, 0.4f, 1.f));

    Renderer renderer;


    // memory layout for helpers
    VertexBufferLayout layout;
    layout.Push<float>(3);
    layout.Push<float>(3);

    // memory layout for pointcloud
    VertexBufferLayout layoutPc;
    layoutPc.Push<float>(3); // xyz
    layoutPc.Push<float>(1); // intensity


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




    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>("/media/michal/ext/skierniewice/cloud.pcd",*cloud);

    const int point_count = cloud->size();
    int len =0;
    auto draw_buffer_vertices_init = pclToBuffer(cloud, len, 1.0f);


    std::unique_ptr<VertexBuffer> vb_points1 = std::make_unique<VertexBuffer>(draw_buffer_vertices_init.get(),len * sizeof(float));
    std::unique_ptr< VertexArray> va_points1 = std::make_unique<VertexArray>();
    va_points1->AddBuffer(*vb_points1, layoutPc);

    Eigen::Vector3f s_max{std::numeric_limits<float>::min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::min()};
    Eigen::Vector3f s_min{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};

    Eigen::Vector3f middle = s_max-s_min;

    for (const auto f: *cloud){
        s_max.x() = std::max(s_max.x(), f.x);
        s_max.y() = std::max(s_max.y(), f.y);
        s_max.z() = std::max(s_max.z(), f.z);

        s_min.x() = std::min(s_min.x(), f.x);
        s_min.y() = std::min(s_min.y(), f.y);
        s_min.z() = std::min(s_min.z(), f.z);

    }


    float cut_z =0;
    while (!glfwWindowShouldClose(window)) {


        GLCall(glEnable(GL_DEPTH_TEST));
        GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO &io = ImGui::GetIO();
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glm::mat4 proj = glm::perspective(30.f, 1.0f * width / height, 0.05f, 1000.0f);

        glm::mat4 model_translate = glm::translate(glm::mat4(1.0f), view_translation);
        glm::mat4 model_rotation_1 = glm::rotate(model_translate, rot_x, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 model_rotation_2 = glm::rotate(model_rotation_1, rot_y, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 model_rotation_3 = glm::rotate(model_rotation_2, (float) (0.5f * glm::pi<float>()), glm::vec3(-1.0f, 0.0f, 0.0f));
        glm::mat4 model_rotation_4 = glm::translate(model_rotation_2, glm::vec3(0.0f, 0.0f, -imgui_depth));

        glm::mat4 scan2_cfg;
        shader.Bind(); // bind shader to apply uniform

        // draw coordinate system
        shader.setUniformMat4f("u_MVP", proj * model_rotation_2);
        renderer.Draw(va, ib, shader, GL_LINES);

        // draw pointcloud
        shader_pc.Bind(); // bind shader to apply uniform
        shader_pc.setUniformMat4f("u_MVPPC", proj * model_rotation_4);
        shader_pc.setUniform4f("u_COLOR", 1, 0, 0, 1);

        renderer.DrawArray(*va_points1, shader_pc, GL_POINTS, point_count);


        ImGui::Begin("Calibration Demo");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

        ImGui::InputFloat("height", &cut_z, 0.1,1);
        if(ImGui::Button("export")){
            float res = 0.1;

            int image_h = 1.1*(s_max.x() - s_min.x())/res;
            int image_w = 1.1*(s_max.y() - s_min.y())/res;
            std::vector<u_int8_t> data;
            data.resize(image_h*image_w,255);
            for (auto &p : *cloud){
                int x = (p.x-s_min.x())/res;
                int y = (p.y-s_min.y())/res;
                if (x > 0 && x < image_w && y > 0 && y< image_h && p.z>cut_z ){
                    data[y*image_w+x]= 0;
                }

            }
            char fn[1024];
            snprintf(fn,1024,"/tmp/floor_%.2f.pgm", cut_z);
            writePGM(fn, data, image_h, image_w);
        }
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